"""synth-intents v5.11 适配: 外部数据仓库的 synth-intents-v5.11/ 材料与绑定 -> 逐题的训练样本, 按训练目标的配比抽题. 只做训练.

格式与检查见外部数据仓库的 synth-intents-v5.11/schema.py: 每个领域一份题库、一份材料清单, 材料上挂绑定 {题, 答案}.
这里一个绑定读成一个 V5Item, 数据目录里的 schema.py 先把每个领域检查一遍, 有问题就报 ValueError.

  选项   模型看到的样子与推理服务相同 (serve.menus.option_row): 键写法「键: 说明」, 说明为 null 只有键;
         是非题 no / yes 两行; 列表写法与分级说明原样. 绑定写了 maskable 的题另存一份只有键的样子 (bare)
  答案   硬标签 -> target None, gold 是那一行; 分布 -> 归一化成 target, gold 是概率最大的一行

抽一道题 (V5Sampler) 分四步: 按当前步的配比挑训练目标 -> 有这个目标的领域里等概率挑一个 ->
这个领域里等概率挑一个题型 -> 等概率挑一份材料 (再在它这个题型的绑定里挑一个). 大领域、材料多的题型不会挤掉别人.
配比 (parse_mix) 可以随训练步数分段变化, 但数据里有的每个目标在每一段都要有份额.

反事实 (v5.5): 材料写了 contrast_of (它改自哪一份材料) 时, 它上面每个绑定记下原材料上同一道题的绑定 (contrast).
抽到这样的改动题, 原题一起出, 两道题进同一批: 两份材料只差那一处关键事实, 梯度只能落在那里.
这一抽算两道题 (批的大小按题数算); 原题自己照常也会被抽到.
contrast_balance (--contrast-balance) 时原题自己被抽到也带上它的改动题 (edit_map), 两边出现次数相同;
不开时原题比改动题多出一份, 而两份标签相反, 交叉熵会偏向原题的答案.

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
import itertools
import os
import pathlib
import random
from dataclasses import dataclass, replace

from sors.core.menu import MenuExample, reorder_menu
from sors.core.prompt import state_text
from sors.serve.menus import option_row
from sors.data.paths import DEFAULT_ROOT, asset_path

DEFAULT_DIR = DEFAULT_ROOT / "synth-intents-v5.11"


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
    qtype: str  # choice | bool | score (分级从低到高, 类 id 就是级别)
    target: list[float] | None  # 分布答案归一化后的各行概率; None = 硬标签
    gold: int
    pair: int | None = None  # 成对的另一半 (原题与它的 other 版, 同一份材料) 在 load_synth_v5 结果里的下标
    fallback_row: str | None = None
    fallback_bare: str | None = None
    fallback_correct: bool = False
    contrast: int | None = None  # 改动题: 原材料 (contrast_of) 上同一道题的绑定下标; 抽到它时原题一起出
    cot: str | None = None  # 绑定的 cot: 程序写的计算过程 (只有 curric 有), 只作训练时的语言模型目标
    categories: tuple[str, ...] = ()  # 能力类别 (数据的 categories/<domain>.json, 多标签; 长材料多一个 long_context); 数据没标就是空
    traces: dict | None = None  # 蒸馏老师读的计算过程 {(说法下标, 问法下标): 文本} (attach_traces); None = 没有


def attach_traces(items: list[V5Item], traces) -> list[V5Item]:
    """把外部计算过程 (每条 {context_id, question_id, text, ask, cot}) 挂到对应绑定上; 对不上的条目忽略."""
    by = {}
    for t in traces:
        by.setdefault((t["context_id"], t["question_id"]), {})[(int(t["text"]), int(t["ask"]))] = t["cot"]
    return [replace(it, traces=by[(it.context_id, it.question_id)]) if (it.context_id, it.question_id) in by else it
            for it in items]


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
                  asks=list(q["ask"]), rows=rows, bare=bare, qtype={"yes_no": "bool", "score": "score"}.get(kind, "choice"),
                  target=target, gold=gold, fallback_row=extra, fallback_bare=extra_bare,
                  fallback_correct=(fallback, b.get("uncertainty")) in
                  (("other", "menu_missing"), ("unknown", "insufficient_evidence")), cot=b.get("cot"))


def load_synth_v5(data_dir=None, *, datasets_dir=None) -> list[V5Item]:
    """全部领域的全部绑定, 领域按名字排序、领域内按材料与绑定的顺序. 格式检查过不了的领域报 ValueError.
    other 版的绑定与同一份材料上原题的绑定互记下标 (pair)."""
    data_dir = pathlib.Path(data_dir) if data_dir is not None else asset_path("synth-intents-v5.11", datasets_dir)
    schema = _schema(data_dir)
    out = []
    for domain in schema.domains(data_dir):
        bank, contexts = schema.load(domain, data_dir)
        errs = schema.problems(domain, bank, contexts)
        if errs:
            raise ValueError(f"{domain}: {len(errs)} format problems, e.g. {errs[:2]}; "
                              f"run {data_dir / 'schema.py'} {domain} to see them all")
        cats = schema.load_categories(domain, data_dir) if hasattr(schema, "load_categories") else None
        if cats is not None:
            errs = schema.category_problems(domain, contexts, cats)
            if errs:
                raise ValueError(f"{domain}: {len(errs)} category problems, e.g. {errs[:2]}; "
                                 f"run {data_dir / 'schema.py'} {domain} to see them all")
        by_id = {q["id"]: q for q in bank}
        for c in contexts:
            start = len(out)
            out += [replace(_item(schema, domain, c, b, by_id[b["question"]]),
                            categories=tuple(schema.categories_of(c, b, cats)) if cats is not None else ())
                    for b in c["questions"]]
            at = {b["question"]: start + j for j, b in enumerate(c["questions"])}
            for j, b in enumerate(c["questions"]):
                base = by_id[b["question"]].get("other_of")
                if base is not None:
                    i, k = start + j, at[base]
                    out[i], out[k] = replace(out[i], pair=k), replace(out[k], pair=i)
        _link_contrasts(out, contexts, domain)
    return out


def drop_categories(items: list[V5Item], excluded: set[str]) -> list[V5Item]:
    """去掉带任一 excluded 类别的绑定 (--exclude-categories). 留下的绑定之间的下标 (pair、contrast) 重新编号,
    指向被去掉的绑定的那一头置空: 原题还在、改动题去掉了, 原题照常单出; 反过来改动题单出.
    excluded 为空时原样返回; 不认识的类别报 ValueError."""
    if not excluded:
        return items
    known = {c for it in items for c in it.categories} | {"long_context"}
    if any(it.categories == () for it in items):
        raise ValueError("--exclude-categories needs data whose every binding has categories")
    unknown = set(excluded) - known
    if unknown:
        raise ValueError(f"unknown categories {sorted(unknown)}; the data has {sorted(known)}")
    keep = [i for i, it in enumerate(items) if not set(excluded) & set(it.categories)]
    new = {old: k for k, old in enumerate(keep)}
    return [replace(items[i], pair=new.get(items[i].pair), contrast=new.get(items[i].contrast)) for i in keep]


def _link_contrasts(out: list[V5Item], contexts: list[dict], domain: str) -> None:
    """改动材料的每个绑定指向原材料 (contrast_of) 上同一道题的绑定. 原材料没有这道题的绑定就不连."""
    at = {(it.context_id, it.question_id): i for i, it in enumerate(out) if it.domain == domain}
    for c in contexts:
        src = c.get("contrast_of")
        if src is None:
            continue
        for b in c["questions"]:
            j = at.get((src, b["question"]))
            if j is not None:
                i = at[(c["id"], b["question"])]
                out[i] = replace(out[i], contrast=j)


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
    trace = partner_trace = None
    if it.traces:
        ai = it.asks.index(question)
        trace = it.traces.get((it.texts.index(query), ai))
        partner_trace = it.traces.get((it.texts.index(other), ai)) if other is not None else None
    ex = MenuExample(query=query, options=list(range(k)), gold_idx=it.gold, label=it.gold, option_names=list(rows),
                     context_label=it.label, question=question, qtype=it.qtype, target=it.target,
                     partner_query=other, cot=it.cot, trace=trace, partner_trace=partner_trace)
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


def draw_of(items: list[V5Item], i: int, edits: dict[int, int] | None = None) -> tuple[tuple[int, ...], ...]:
    """抽到绑定 i 时出的题: 一个单元 (unit_of), 改动题再加上原题的单元. 每个单元出一道题.
    edits (edit_map) 给了就是曝光平衡: 原题的这一抽也带上它的改动题, 两边出现次数相同."""
    u = unit_of(items, i)
    src = next((items[k].contrast for k in u if items[k].contrast is not None), None)
    if src is not None:
        return (u, unit_of(items, src))
    edit = next((edits[k] for k in u if k in edits), None) if edits else None
    return (u,) if edit is None else (u, unit_of(items, edit))


def edit_map(items: list[V5Item]) -> dict[int, int]:
    """原题 -> 它的改动题 (下标). 一份原材料有几份改动时取下标最小的那份, 其余改动题照常只在自己被抽到时带上原题."""
    out: dict[int, int] = {}
    for i, it in enumerate(items):
        if it.contrast is not None:
            out.setdefault(it.contrast, i)
    return out


_GROUPS = itertools.count()


def draw_examples(items: list[V5Item], draw: tuple[tuple[int, ...], ...], rng: random.Random, mask_rate: float,
                  fallback_rate: float) -> list[MenuExample]:
    """一抽出的题. 原题与改动题成对出时两道题记同一个 contrast_group (进程内递增, 不消耗随机数), 成对损失按它配对."""
    out = [ex for u in draw for ex in unit_examples(items, u, rng, mask_rate, fallback_rate)]
    if len(draw) == 2:
        g = next(_GROUPS)
        out = [replace(ex, contrast_group=g) for ex in out]
    return out


class V5Sampler:
    """训练批里 v5 的那一份. 每调用一次算训练的一步 (train() 每步调一次 sample_fn), 配比按这个步数取.
    rate 是遮说明的比例 (只作用在绑定写了 maskable 的题上)."""

    def __init__(self, items: list[V5Item], mix: Mix, mask_rate: float = 0.0, fallback_rate: float = 0.5,
                 contrast_balance: bool = False):
        _check_fallback_rate(fallback_rate)
        self.fallback_rate = fallback_rate
        self.items, self.mix, self.mask_rate, self.step = items, mix, mask_rate, 0
        self.edits = edit_map(items) if contrast_balance else None
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
        while len(out) < n:
            for g in rng.choices(goals, [weights[x] for x in goals], k=n - len(out)):
                ctxs = rng.choice(rng.choice(self.tree[g]))
                draw = draw_of(self.items, rng.choice(rng.choice(ctxs)), self.edits)
                if len(draw) > n - len(out):
                    continue  # 改动题连原题要两个位置, 只剩一个: 这一抽作废, 外层再补
                out += draw_examples(self.items, draw, rng, self.mask_rate, self.fallback_rate)
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
    成对的两半 (原题与 other 版) 算一个单元、随机选一版，同 V5Sampler。改动题一抽连原题出两道 (见 draw_of)。
    round_size 是一轮出的题数 (改动题算两道)。"""

    def __init__(self, items: list[V5Item], passes: dict[str, int], mask_rate: float = 0.0, fallback_rate: float = 0.5,
                 contrast_balance: bool = False):
        _check_fallback_rate(fallback_rate)
        self.fallback_rate = fallback_rate
        missing = {it.goal for it in items} ^ set(passes)
        if missing:
            raise ValueError(f"the passes and the data disagree on goals {sorted(missing)}")
        self.items, self.passes, self.mask_rate = items, passes, mask_rate
        edits = edit_map(items) if contrast_balance else None
        draws = sorted({draw_of(items, i, edits) for i in range(len(items))})
        self.units = [d for d in draws for _ in range(passes[items[d[0][0]].goal])]
        self.round_size = sum(len(d) for d in self.units)
        self.contrast_draws = sum(len(d) == 2 for d in self.units)  # 一轮里成对出的抽取数 (改动题带原题, 平衡时也有原题带改动题)
        self.queue: list[tuple[tuple[int, ...], ...]] = []
        self.rounds = 0  # 已经开了几轮

    def __call__(self, n: int, rng: random.Random) -> list[MenuExample]:
        out = []
        while len(out) < n:
            if not self.queue:
                self.queue = rng.sample(self.units, len(self.units))[::-1]  # 从尾部 pop, 顺序仍是均匀的
                self.rounds += 1
            if len(self.queue[-1]) > n - len(out):
                # 改动题连原题要两个位置, 这批只剩一个: 换一道单出的题先出, 改动题留在队尾给下一批
                j = next((j for j in range(len(self.queue) - 2, -1, -1) if len(self.queue[j]) == 1), None)
                if j is None:
                    out += draw_examples(self.items, self.queue.pop()[:1], rng, self.mask_rate, self.fallback_rate)
                    continue
                self.queue[j], self.queue[-1] = self.queue[-1], self.queue[j]
            out += draw_examples(self.items, self.queue.pop(), rng, self.mask_rate, self.fallback_rate)
        return out


# ---------------------------------------------------------------- 分层抽批

def batch_quota(shares: dict[str, float], n: int) -> dict[str, int]:
    """n 个位置按 shares 的比例分给各目标 (最大余数法), 每个目标至少一个; 不够每个目标一个时报 ValueError.
    分到 0 的目标从「分到的减应得」最多的目标那里拿一个."""
    if n < len(shares):
        raise ValueError(f"{n} slots cannot give each of {len(shares)} goals one")
    total = sum(shares.values())
    exact = {g: n * v / total for g, v in shares.items()}
    alloc = {g: int(x) for g, x in exact.items()}
    for g in sorted(exact, key=lambda g: (alloc[g] - exact[g], g))[: n - sum(alloc.values())]:
        alloc[g] += 1
    for g in sorted(g for g in alloc if alloc[g] == 0):
        donor = max((d for d in alloc if alloc[d] > 1), key=lambda d: (alloc[d] - exact[d], d))
        alloc[donor] -= 1
        alloc[g] = 1
    return alloc


class V5Strata:
    """分层抽批 (--contrast-pairs): 每批固定 pairs 对原题+改动题, 其余 batch_size - 2·pairs 个位置按目标固定份额
    (batch_quota, 份额正比于该目标一轮里单出的题数). 原题与改动题总是成对出 (曝光相同, 同 contrast_balance),
    同一对只算一个单元. 每个目标、以及成对的单元各有自己的队列, 各自按轮打乱取完再开下一轮; 遍数同 V5Rounds.
    目的: 每一步的批构成相同, 成对损失每步都有、量也不变, 不会某几步一下子来很多对."""

    def __init__(self, items: list[V5Item], passes: dict[str, int], pairs: int, batch_size: int,
                 mask_rate: float = 0.0, fallback_rate: float = 0.5):
        _check_fallback_rate(fallback_rate)
        missing = {it.goal for it in items} ^ set(passes)
        if missing:
            raise ValueError(f"the passes and the data disagree on goals {sorted(missing)}")
        if pairs < 1 or 2 * pairs >= batch_size:
            raise ValueError(f"--contrast-pairs {pairs} must be >= 1 and leave room in a batch of {batch_size}")
        self.items, self.passes, self.mask_rate, self.fallback_rate = items, passes, mask_rate, fallback_rate
        self.pairs, self.batch_size = pairs, batch_size
        edits = edit_map(items)
        draws = {draw_of(items, i, edits) for i in range(len(items))}
        goal = lambda d: max(passes[items[k].goal] for u in d for k in u)  # noqa: E731
        self.pair_units = sorted({tuple(sorted(d)) for d in draws if len(d) == 2})
        singles: dict[str, list] = {}
        for d in sorted(d for d in draws if len(d) == 1):
            singles.setdefault(items[d[0][0]].goal, []).extend([d] * passes[items[d[0][0]].goal])
        self.singles = singles
        self.pool = [d for d in self.pair_units for _ in range(goal(d))]
        self.quota = batch_quota({g: len(v) for g, v in singles.items()}, batch_size - 2 * pairs)
        self.queues: dict[str, list] = {}

    def _take(self, key: str, pool: list, k: int, rng: random.Random) -> list:
        out = []
        while len(out) < k:
            q = self.queues.get(key)
            if not q:
                q = self.queues[key] = rng.sample(pool, len(pool))[::-1]
            out.append(q.pop())
        return out

    def __call__(self, n: int, rng: random.Random) -> list[MenuExample]:
        if n != self.batch_size:
            raise ValueError(f"the strata were planned for batches of {self.batch_size}, asked for {n}")
        draws = self._take("\0pairs", self.pool, self.pairs, rng)
        for g in sorted(self.quota):
            draws += self._take(g, self.singles[g], self.quota[g], rng)
        return [ex for d in draws for ex in draw_examples(self.items, d, rng, self.mask_rate, self.fallback_rate)]


# ---------------------------------------------------------------- 按题型分段抽批

QTYPE_ORDER = ("choice", "bool", "score")


def parse_type_quota(spec: str, qtypes: set[str]) -> dict[str, int]:
    """'choice=18,bool=14,score=4' -> {题型: 每批题数}. 数据里有的每种题型都要列出, 题数 >= 1; 不认识的题型报错."""
    out: dict[str, int] = {}
    for part in spec.split(","):
        name, _, value = part.partition("=")
        name = name.strip()
        if name not in QTYPE_ORDER or name in out:
            raise ValueError(f"--type-quota: unknown or repeated question type {name!r} (known: {', '.join(QTYPE_ORDER)})")
        try:
            out[name] = int(value)
        except ValueError:
            raise ValueError(f"--type-quota: {part!r} is not type=count") from None
        if out[name] < 1:
            raise ValueError(f"--type-quota: {name} needs at least one question per batch, got {out[name]}")
    if set(out) != qtypes:
        raise ValueError(f"--type-quota lists {sorted(out)} but the data has {sorted(qtypes)}")
    return out


class V5TypeStrata:
    """按题型分段抽批 (--type-quota): 每批选择题 / 是非题 / 评分题各 quota[t] 道, 三段的题各自按自己的 loss 算.
    分段按绑定本来的题型 (V5Item.qtype); 兜底增强把是非题变成三行菜单时, 出来的题记 choice, 按选择题算 loss.
    每种题型一个队列, 一轮把这种题型的绑定按遍数 (parse_passes) 各出一遍, 打乱, 取完再开下一轮.
    原题与改动题总是成对出 (同 contrast_balance, 同一对只算一个单元), 一对占自己题型的两个位置;
    这一段只剩一个位置时换队里下一道单出的题先出, 没有单出的题就只出这一对的前一半."""

    def __init__(self, items: list[V5Item], passes: dict[str, int], quota: dict[str, int],
                 mask_rate: float = 0.0, fallback_rate: float = 0.5):
        _check_fallback_rate(fallback_rate)
        missing = {it.goal for it in items} ^ set(passes)
        if missing:
            raise ValueError(f"the passes and the data disagree on goals {sorted(missing)}")
        if set(quota) != {it.qtype for it in items}:
            raise ValueError(f"the quota lists {sorted(quota)} but the data has {sorted({it.qtype for it in items})}")
        self.items, self.passes, self.quota = items, passes, dict(quota)
        self.mask_rate, self.fallback_rate = mask_rate, fallback_rate
        self.batch_size = sum(quota.values())
        edits = edit_map(items)
        draws = sorted({tuple(sorted(d)) if len(d) == 2 else d for d in (draw_of(items, i, edits) for i in range(len(items)))})
        self.pools: dict[str, list] = {t: [] for t in quota}
        for d in draws:
            first = items[d[0][0]]
            reps = max(passes[items[k].goal] for u in d for k in u)
            self.pools[first.qtype].extend([d] * reps)
        self.pair_units = sum(len(d) == 2 for d in draws)
        self.queues: dict[str, list] = {}
        self.rounds = {t: 0 for t in quota}

    def _segment(self, t: str, rng: random.Random) -> list:
        out, room = [], self.quota[t]
        while room:
            q = self.queues.get(t)
            if not q:
                q = self.queues[t] = rng.sample(self.pools[t], len(self.pools[t]))[::-1]
                self.rounds[t] += 1
            if len(q[-1]) > room:
                j = next((j for j in range(len(q) - 2, -1, -1) if len(q[j]) <= room), None)
                if j is None:
                    out.append(q.pop()[:1])
                    room -= 1
                    continue
                q[j], q[-1] = q[-1], q[j]
            d = q.pop()
            out.append(d)
            room -= len(d)
        return out

    def __call__(self, n: int, rng: random.Random) -> list[MenuExample]:
        if n != self.batch_size:
            raise ValueError(f"the type quota {self.quota} makes batches of {self.batch_size}, asked for {n}")
        draws = [d for t in QTYPE_ORDER if t in self.quota for d in self._segment(t, rng)]
        return [ex for d in draws for ex in draw_examples(self.items, d, rng, self.mask_rate, self.fallback_rate)]
