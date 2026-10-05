"""synth-intents v5.3 适配: 外部数据仓库的 synth-intents-v5.3/ 材料与绑定 -> 逐题的训练样本, 按训练目标的配比抽题. 只做训练.

格式与检查见外部数据仓库的 synth-intents-v5.3/schema.py: 每个领域一份题库、一份材料清单, 材料上挂绑定 {题, 答案}.
这里一个绑定读成一个 V5Item, 数据目录里的 schema.py 先把每个领域检查一遍, 有问题就报 ValueError.

  选项   模型看到的样子与推理服务相同 (serve.menus.option_row): 键写法「键: 说明」, 说明为 null 只有键;
         是非题 no / yes 两行; 列表写法与分级说明原样. 绑定写了 maskable 的题另存一份只有键的样子 (bare)
  答案   硬标签 -> target None, gold 是那一行; 分布 -> 归一化成 target, gold 是概率最大的一行

抽一道题 (V5Sampler) 分四步: 按当前步的配比挑训练目标 -> 有这个目标的领域里等概率挑一个 ->
这个领域里等概率挑一个题型 -> 等概率挑一份材料 (再在它这个题型的绑定里挑一个). 大领域、材料多的题型不会挤掉别人.
配比 (parse_mix) 可以随训练步数分段变化, 但数据里有的每个目标在每一段都要有份额.

另一种抽法是按轮抽 (V5Rounds): 一轮把每个绑定出一遍, parse_passes 给了遍数的目标出那么多遍, 轮内整体打乱,
一轮取完再打乱出下一轮. 用来保证训练里每个绑定至少见过一次; 各目标的份额就是它们在数据里的量乘遍数.

出成样本 (item_example): 随机挑一种问法; 材料有几种说法时挑两种不同的, 一种当上下文, 另一种放进 partner_query
(开一致性配对时第二份读它); 按比例遮掉说明 (只对 maskable 的题, 两份跟着同一次抽签); 最后打乱行序.

兜底增强: 绑定通过 fallback 标记允许添加 Other 或信息不足选项，uncertainty 决定新增项是否为答案。
fallback_rate 默认 0.5，每次抽取出一个菜单版本。既有 other_of 配对也按该概率选择原版或 Other 版。
诊断用的 pair_examples 可一次返回两版。--consistency 的第二份复用已经选定的菜单与目标，另换措辞和行序。
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import random
from dataclasses import dataclass, replace

from sors.core.menu import MenuExample, reorder_menu
from sors.core.prompt import state_text
from sors.serve.menus import option_row
from sors.data.paths import DEFAULT_ROOT, asset_path

DEFAULT_DIR = DEFAULT_ROOT / "synth-intents-v5.3"


@dataclass(frozen=True)
class V5Item:
    domain: str
    context_id: str
    question_id: str
    kind: str  # 题型, 抽题时领域内按它均分
    goal: str  # 训练目标, 绑定上写了就用绑定的, 否则用材料的
    label: str  # 上下文标题
    texts: list[str]  # 材料的几种说法, 已写成提示里的样子 (JSON 展开成缩进 2 格)
    asks: list[str]
    rows: list[str]  # 菜单各行, 带说明
    bare: list[str] | None  # 遮掉说明后的各行; None = 这个绑定不许遮
    qtype: str  # choice | bool
    target: list[float] | None  # 分布答案归一化后的各行概率; None = 硬标签
    gold: int
    pair: int | None = None  # 成对的另一半 (原题与它的 other 版, 同一份材料) 在 load_synth_v5 结果里的下标
    fallback_row: str | None = None
    fallback_bare: str | None = None
    fallback_correct: bool = False


def _schema(data_dir: pathlib.Path):
    spec = importlib.util.spec_from_file_location(f"synth_v5_schema_{abs(hash(str(data_dir)))}", data_dir / "schema.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    if "JEVBENCH_DIR" not in os.environ and hasattr(m, "JEVBENCH"):
        m.JEVBENCH = pathlib.Path(__file__).resolve().parents[3] / "data" / "jevbench"
    return m


def _item(schema, domain: str, c: dict, b: dict, q: dict) -> V5Item:
    kind = schema.option_kind(q)
    if kind in ("keyed", "yes_no"):
        rows = [option_row(k, d) for k, d in q["options"].items()]
        bare = [option_row(k, None) for k in q["options"]] if b.get("maskable") else None
    else:
        rows, bare = [option_row(s, None) for s in q["options"]], None
    names = schema.option_names(q)
    a = b["answer"]
    if isinstance(a, dict):
        w = [float(a.get(n, 0)) for n in names]
        target = [x / sum(w) for x in w]
        gold = max(range(len(w)), key=w.__getitem__)
    else:
        target, gold = None, (a if kind == "score" else names.index(a))
    fallback = b.get("fallback")
    extra, extra_bare = None, None
    if fallback:
        key, description = schema.FALLBACKS[fallback]
        extra = option_row(key, description) if kind in ("keyed", "yes_no") else description
        extra_bare = key
    return V5Item(domain=domain, context_id=c["id"], question_id=q["id"], kind=q.get("kind", q["id"]),
                  goal=schema.goal_of(c, b), label=c["label"], texts=[state_text(t) for t in schema.variants(c["text"])],
                  asks=list(q["ask"]), rows=rows, bare=bare, qtype="bool" if kind == "yes_no" else "choice",
                  target=target, gold=gold, fallback_row=extra, fallback_bare=extra_bare,
                  fallback_correct=(fallback, b.get("uncertainty")) in
                  (("other", "menu_missing"), ("unknown", "insufficient_evidence")))


def load_synth_v5(data_dir=None, *, datasets_dir=None) -> list[V5Item]:
    """全部领域的全部绑定, 领域按名字排序、领域内按材料与绑定的顺序. 格式检查过不了的领域报 ValueError.
    other 版的绑定与同一份材料上原题的绑定互记下标 (pair)."""
    data_dir = pathlib.Path(data_dir) if data_dir is not None else asset_path("synth-intents-v5.3", datasets_dir)
    schema = _schema(data_dir)
    out = []
    for domain in schema.domains(data_dir):
        bank, contexts = schema.load(domain, data_dir)
        errs = schema.problems(domain, bank, contexts)
        if errs:
            raise ValueError(f"{domain}: {len(errs)} format problems, e.g. {errs[:2]}; "
                              f"run {data_dir / 'schema.py'} {domain} to see them all")
        by_id = {q["id"]: q for q in bank}
        for c in contexts:
            start = len(out)
            out += [_item(schema, domain, c, b, by_id[b["question"]]) for b in c["questions"]]
            at = {b["question"]: start + j for j, b in enumerate(c["questions"])}
            for j, b in enumerate(c["questions"]):
                base = by_id[b["question"]].get("other_of")
                if base is not None:
                    i, k = start + j, at[base]
                    out[i], out[k] = replace(out[i], pair=k), replace(out[k], pair=i)
    return out


def _choose(it: V5Item, rng: random.Random, mask_rate: float) -> tuple[bool, str, str | None]:
    """遮不遮 (只在 maskable 且 rate > 0 时抽) -> 说法 (有几种时挑两种, 第二种给一致性配对的第二份)."""
    masked = it.bare is not None and mask_rate > 0 and rng.random() < mask_rate
    if len(it.texts) > 1:
        query, other = rng.sample(it.texts, 2)
    else:
        query, other = it.texts[0], None
    return masked, query, other


def _example(it: V5Item, masked: bool, query: str, other: str | None, question: str, rng: random.Random) -> MenuExample:
    rows = it.bare if masked else it.rows
    k = len(rows)
    ex = MenuExample(query=query, options=list(range(k)), gold_idx=it.gold, label=it.gold, option_names=list(rows),
                     context_label=it.label, question=question, qtype=it.qtype, target=it.target,
                     partner_query=other)
    return reorder_menu(ex, rng.sample(range(k), k))


def _check_fallback_rate(rate: float) -> None:
    if not 0 <= rate <= 1:
        raise ValueError(f"fallback_rate must be a probability in 0..1, got {rate}")


def _add_fallback(it: V5Item) -> V5Item:
    """Create one augmented view without modifying the stored source item."""
    k = len(it.rows)
    if k >= 256:
        raise ValueError("fallback augmentation needs a free menu slot (maximum 256)")
    target = None if it.fallback_correct or it.target is None else [*it.target, 0.0]
    return replace(it, rows=[*it.rows, it.fallback_row],
                   bare=[*it.bare, it.fallback_bare] if it.bare is not None else None,
                   gold=k if it.fallback_correct else it.gold, target=target, qtype="choice")


def item_example(it: V5Item, rng: random.Random, mask_rate: float, fallback_rate: float = 0.5) -> MenuExample:
    """Choose a fallback view, then descriptions, phrasings, question wording and row order."""
    _check_fallback_rate(fallback_rate)
    if it.fallback_row is not None and fallback_rate > 0 and (fallback_rate == 1 or rng.random() < fallback_rate):
        it = _add_fallback(it)
    masked, query, other = _choose(it, rng, mask_rate)
    return _example(it, masked, query, other, rng.choice(it.asks), rng)


def pair_examples(a: V5Item, b: V5Item, rng: random.Random, mask_rate: float) -> list[MenuExample]:
    """成对的两个绑定 (原题与 other 版) 出两道题: 遮不遮、说法、问法抽一次两道共用, 行序各打乱各的."""
    masked, query, other = _choose(a, rng, mask_rate)
    question = rng.choice(a.asks)
    return [_example(it, masked, query, other, question, rng) for it in (a, b)]


# ---------------------------------------------------------------- 目标配比

Mix = list[tuple[int, dict[str, float]]]  # [(从第几步起, {目标: 权重})], 按步数升序, 第一段从 0 起


def parse_mix(spec: str | None, goals: set[str]) -> Mix:
    """--mix 的值. 不给 = 数据里的每个目标同样多. 写法:
         long_menu=4,breadth=2,complex=1                      全程一个配比
         0: long_menu=6,breadth=2; 1000: long_menu=3,...      分段, 到第 1000 步换成后一个
    权重是相对值, 要是正数. 数据里有的目标每一段都要写 (每种目标从头到尾都保留份额); 写了数据里没有的目标报错."""
    if not spec:
        return [(0, {g: 1.0 for g in sorted(goals)})]
    stages = []
    for part in (p.strip() for p in spec.split(";")):
        if not part:
            continue
        start, sep, body = part.partition(":")
        if not sep:
            start, body = "0", part
        try:
            step = int(start)
            weights = {g.strip(): float(w) for g, w in (e.split("=") for e in body.split(","))}
        except ValueError as e:
            raise ValueError(f"--mix: cannot read {part!r} (write goal=weight,goal=weight, optionally after 'step:')") from e
        stages.append((step, weights))
    steps = [s for s, _ in stages]
    if steps != sorted(set(steps)) or steps[0] != 0:
        raise ValueError(f"--mix: stages must start at step 0 and go up, got steps {steps}")
    for step, weights in stages:
        extra = sorted(set(weights) - goals)
        if extra:
            raise ValueError(f"--mix: no data for goal(s) {extra}; the data has {sorted(goals)}")
        missing = sorted(g for g in goals if weights.get(g, 0) <= 0)
        if missing or any(w <= 0 for w in weights.values()):
            raise ValueError(f"--mix: every goal in the data keeps a positive share in every stage; "
                             f"the stage from step {step} leaves out {missing or weights}")
    return stages


def mix_at(stages: Mix, step: int) -> dict[str, float]:
    return next(w for s, w in reversed(stages) if s <= step)


class V5Sampler:
    """训练批里 v5 的那一份. 每调用一次算训练的一步 (train() 每步调一次 sample_fn), 配比按这个步数取.
    rate 是遮说明的比例 (只作用在绑定写了 maskable 的题上)."""

    def __init__(self, items: list[V5Item], mix: Mix, mask_rate: float = 0.0, fallback_rate: float = 0.5):
        _check_fallback_rate(fallback_rate)
        self.fallback_rate = fallback_rate
        self.items, self.mix, self.mask_rate, self.step = items, mix, mask_rate, 0
        tree: dict = {}
        for i, it in enumerate(items):
            if it.pair is not None and i > it.pair:
                continue
            ctx = tree.setdefault(it.goal, {}).setdefault(it.domain, {}).setdefault(it.kind, {})
            ctx.setdefault(it.context_id, []).append(i)
        # 目标 -> [领域 -> [题型 -> [材料 -> [绑定下标]]]], 各层按名字排序, 抽样与 dict 的插入顺序无关
        self.tree = {g: [[list(ks[k].values()) for k in sorted(ks)] for _, ks in sorted(ds.items())]
                     for g, ds in tree.items()}
        missing = set(mix_at(mix, 0)) ^ set(self.tree)
        if missing:
            raise ValueError(f"the mix and the data disagree on goals {sorted(missing)}")

    def __call__(self, n: int, rng: random.Random) -> list[MenuExample]:
        """n 次抽取出 n 道题；每个原题/Other 单元随机选择一个菜单版本。"""
        self.step += 1
        weights = mix_at(self.mix, self.step)
        goals = sorted(weights)
        out = []
        for g in rng.choices(goals, [weights[x] for x in goals], k=n):
            ctxs = rng.choice(rng.choice(self.tree[g]))
            out += unit_examples(self.items, unit_of(self.items, rng.choice(rng.choice(ctxs))), rng, self.mask_rate,
                                 self.fallback_rate)
        return out


def unit_of(items: list[V5Item], i: int) -> tuple[int, ...]:
    """一次抽取出的绑定下标: 单个绑定是它自己; 成对的是 (原题, other 版), 按下标排, 原题在前."""
    p = items[i].pair
    return (i,) if p is None else tuple(sorted((i, p)))


def unit_examples(items: list[V5Item], unit: tuple[int, ...], rng: random.Random, mask_rate: float,
                  fallback_rate: float = 0.5) -> list[MenuExample]:
    _check_fallback_rate(fallback_rate)
    if len(unit) == 1:
        return [item_example(items[unit[0]], rng, mask_rate, fallback_rate)]
    base, other = sorted((items[i] for i in unit), key=lambda it: len(it.rows))
    augmented = fallback_rate > 0 and (fallback_rate == 1 or rng.random() < fallback_rate)
    return [item_example(other if augmented else base, rng, mask_rate, fallback_rate=0.0)]


# ---------------------------------------------------------------- 按轮抽

def parse_passes(spec: str | None, goals: set[str]) -> dict[str, int]:
    """--passes 的值: 一轮里各目标的绑定各出几遍. 不写的目标 1 遍. 写法 complex=3,edge_case=3,long_context=3;
    遍数是正整数, 写了数据里没有的目标报错."""
    out = {g: 1 for g in sorted(goals)}
    for e in (x.strip() for x in (spec or "").split(",")):
        if not e:
            continue
        g, sep, v = e.partition("=")
        g = g.strip()
        if not sep or not v.strip().isdigit() or int(v) < 1:
            raise ValueError(f"--passes: cannot read {e!r} (write goal=N with N a whole number >= 1)")
        if g not in goals:
            raise ValueError(f"--passes: no data for goal {g!r}; the data has {sorted(goals)}")
        out[g] = int(v)
    return out


class V5Rounds:
    """按轮抽: 一轮把每个绑定出一遍, parse_passes 列了遍数的目标出那么多遍, 轮内整体打乱.
    一批从上一批停下的地方接着取, 一轮取完就打乱出下一轮, 一批可以跨过两轮的边界.
    成对的两半 (原题与 other 版) 算一个单元、随机选一版，同 V5Sampler。round_size 是一轮的抽取次数。"""

    def __init__(self, items: list[V5Item], passes: dict[str, int], mask_rate: float = 0.0, fallback_rate: float = 0.5):
        _check_fallback_rate(fallback_rate)
        self.fallback_rate = fallback_rate
        missing = {it.goal for it in items} ^ set(passes)
        if missing:
            raise ValueError(f"the passes and the data disagree on goals {sorted(missing)}")
        self.items, self.passes, self.mask_rate = items, passes, mask_rate
        units = sorted({unit_of(items, i) for i in range(len(items))})
        self.units = [u for u in units for _ in range(passes[items[u[0]].goal])]
        self.round_size = len(self.units)
        self.queue: list[tuple[int, ...]] = []
        self.rounds = 0  # 已经开了几轮

    def __call__(self, n: int, rng: random.Random) -> list[MenuExample]:
        out = []
        for _ in range(n):
            if not self.queue:
                self.queue = rng.sample(self.units, len(self.units))[::-1]  # 从尾部 pop, 顺序仍是均匀的
                self.rounds += 1
            out += unit_examples(self.items, self.queue.pop(), rng, self.mask_rate, self.fallback_rate)
        return out
