"""菜单样本的采样: 类切分、组菜单、批量生成. 纯函数, 不碰文本与 tokenizer.

一条样本 = 上下文 (query) + 可选的问句 + 一个选项菜单 (类 id 的有序列表) + 正确选项在菜单里的位置.
菜单每次都重新随机组, 同一个类在不同样本里落到不同位置 —— 模型学的是
「在给出的选项里指出匹配的那个」, 而非「某个类永远对应某个槽」.

LabeledSet 是数据集适配层交上来的统一形状: 一列上下文、一列类 id、类名表,
以及这个数据集里上下文叫什么 (Customer message / Passage) 和每条各自的问句 (可无).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace

from sors.core.tokens import N_SLOTS


@dataclass(frozen=True)
class ClassSplit:
    train: list[int]
    held_out: list[int]


def class_split(n_classes: int, n_held_out: int, seed: int) -> ClassSplit:
    """把 0..n-1 随机切成 train / held_out. held_out 的类在训练里完全不出现,
    用来测「学到的是方法还是记住了标签」."""
    ids = list(range(n_classes))
    random.Random(seed).shuffle(ids)
    return ClassSplit(train=sorted(ids[n_held_out:]), held_out=sorted(ids[:n_held_out]))


@dataclass(frozen=True)
class MenuExample:
    query: str  # 上下文: 用户句 / passage
    options: list[int]  # 类 id, 顺序就是菜单顺序
    gold_idx: int  # 正确选项在 options 里的位置
    label: int  # 正确选项的类 id (== options[gold_idx])
    option_names: list[str]  # 与 options 平行, 菜单里显示的名字
    context_label: str = "Customer message"  # 上下文前面的标签
    question: str | None = None  # 这条样本的问句; None 表示数据集用固定的默认问句
    qtype: str = "choice"  # choice | bool | score, 见 tokens.QTYPES
    codes: list[int] | None = None  # 与 options 平行, 第 i 项绑 <|D{codes[i]}|>; None = 按位置 D0, D1, ...
    target: list[float] | None = None  # 与 options 平行, 各行的目标概率 (软标签); None = 只认 gold_idx 那一行
    partner_query: str | None = None  # 同一份材料的另一种说法; 一致性配对的第二份读它 (partner), None = 两份读同一份

    def __post_init__(self):
        # D 槽只有 N_SLOTS 个. 把菜单压到这个数以内是各数据集管线的责任, 压不住就在抽样当下报错.
        if len(self.options) > N_SLOTS:
            raise ValueError(f"menu has {len(self.options)} options, only {N_SLOTS} D slots; "
                             "the dataset pipeline must cap it")
        if self.codes is not None:
            c = self.codes
            if len(c) != len(self.options) or len(set(c)) != len(c) or not all(0 <= x < N_SLOTS for x in c):
                raise ValueError(f"codes must be {len(self.options)} distinct slots in 0..{N_SLOTS - 1}, got {c}")
        if self.target is not None:
            t = self.target
            if len(t) != len(self.options) or min(t) < 0 or abs(sum(t) - 1) > 1e-6:
                raise ValueError(f"target must be {len(self.options)} non-negative probabilities summing to 1, got {t}")

    @property
    def slot_codes(self) -> list[int]:
        """菜单第 i 项绑的 D 码. 正确答案是 slot_codes[gold_idx]."""
        return list(range(len(self.options))) if self.codes is None else list(self.codes)


def menu_k_range(k_min: int | None, k_max: int) -> tuple[int, int]:
    """k_max 是菜单最多几项. k_min 不给 = 全量: 每个菜单都取 k_max, 池子不够 k_max 就整个池子放进去
    (compose_menu 负责夹). 给了 k_min 才在 k_min..k_max 之间随机抽长度."""
    return (k_max, k_max) if k_min is None else (k_min, k_max)


def draw_k(k_range: tuple[int, int], rng: random.Random, log: bool = False) -> int:
    """菜单长度. log=True 按对数均匀取: 2..256 之间一半落在 ~23 以内, 少数拉到两百多,
    每个槽都轮得到而平均提示长度不爆."""
    lo, hi = k_range
    if not log or lo >= hi:
        return rng.randint(lo, hi)
    return min(hi, max(lo, round(math.exp(rng.uniform(math.log(lo), math.log(hi))))))


def compose_menu(gold: int, pool: list[int], k: int, rng: random.Random) -> tuple[list[int], int]:
    """从 pool 里抽 k-1 个干扰项加上 gold, 打乱. 返回 (options, gold 的位置).
    k 大于 pool 大小时取整个 pool."""
    others = [c for c in pool if c != gold]
    k = min(k, len(others) + 1)
    opts = rng.sample(others, k - 1) + [gold]
    rng.shuffle(opts)
    return opts, opts.index(gold)


class RandomCodes:
    """训练抽题的最后一步: 每道题以概率 rate 换一套码, 其余保持 D0, D1, ... 连续编号. 二元题 (no / yes) 也一样换:
    模型要靠描述答题, 两项菜单写成 D0 / D1 还是别的码都不该有差别. 菜单第 i 项写成这套里的第 i 个,
    答案是正确那一项旁边写的码, 与它排第几行无关.

    设计目的是光看编号推不出答案: 一个码出现在 k 项菜单上时, 它是答案的概率必须是 1/k, 对哪个码都一样.
    所以换码时不单独给答案挑码, 而是整套一起挑: 取到目前为止上菜单次数最少的 k 个码 (并列时随机),
    再随机排进菜单 —— 答案落在这套里的哪一个完全随机. 均衡的是上菜单的次数, 当答案的次数随之期望相同.
    计数覆盖所有题, 连续编号那部分也算 (它们只占 D0..D(k-1)), 跨调用保留 —— 一个 run 用一个实例.
    连续编号的菜单太多时补不齐: 60 项菜单下 rate 低于 0.77, D0..D59 光靠连续编号上菜单的次数就超过均分,
    换码的菜单全用 D60 以后的码, D0..D59 仍然偏多, 但每个码是答案的概率依旧是 1/k.

    rate 0 时原样返回, 不从 rng 取数 —— 不开这个功能的 run 抽题序列与以前逐条相同."""

    def __init__(self, rate: float):
        self.rate = rate
        self.on_menu = [0] * N_SLOTS

    def __call__(self, examples: list[MenuExample], rng: random.Random) -> list[MenuExample]:
        if self.rate <= 0:
            return examples
        out = []
        for ex in examples:
            if rng.random() < self.rate:
                ex = replace(ex, codes=self._codes(len(ex.options), rng))
            for c in ex.slot_codes:
                self.on_menu[c] += 1
            out.append(ex)
        return out

    def _codes(self, k: int, rng: random.Random) -> list[int]:
        least = sorted(range(N_SLOTS), key=lambda c: (self.on_menu[c], rng.random()))[:k]
        rng.shuffle(least)
        return least


# 不变性诊断的菜单变体 (scripts/eval-invariance.py): 同一道题只改一个变量, 看选中的描述变不变.
# 描述带着自己的名字和类 id 走, 比较时按类 id 对齐, 与它落在哪一行、写成哪个码无关.


def reorder_menu(ex: MenuExample, rows: list[int], codes: list[int] | None = None) -> MenuExample:
    """按 rows (原菜单的行号, 顺序即新顺序) 重排或取子集; 正确那一行必须在其中, gold_idx 跟到它的新位置.
    codes 给新菜单每一行绑的码, None = 连续编号 D0, D1, ... 有软标签时各行的目标概率跟着行走, 取子集就在留下的行上重新归一."""
    if len(set(rows)) != len(rows) or not all(0 <= r < len(ex.options) for r in rows):
        raise ValueError(f"rows must be distinct rows of a {len(ex.options)}-option menu, got {rows}")
    if ex.gold_idx not in rows:
        raise ValueError(f"the gold row {ex.gold_idx} must stay on the menu")
    target = None
    if ex.target is not None:
        kept = [ex.target[r] for r in rows]
        target = [x / sum(kept) for x in kept]
    return replace(ex, options=[ex.options[r] for r in rows], option_names=[ex.option_names[r] for r in rows],
                   gold_idx=rows.index(ex.gold_idx), codes=codes, target=target)


def shuffled_rows(ex: MenuExample, rng: random.Random, keep_codes: bool) -> MenuExample:
    """行打乱. keep_codes=False: 仍连续编号, 行和码一起变 (部署形态);
    True: 每条描述保留原来的码, 只有行变 —— 码因此不再按顺序排列."""
    rows = rng.sample(range(len(ex.options)), len(ex.options))
    return reorder_menu(ex, rows, [ex.slot_codes[r] for r in rows] if keep_codes else None)


def reassigned_codes(ex: MenuExample, rng: random.Random) -> MenuExample:
    """行顺序不变, 从 N_SLOTS 个码里随机挑 k 个重新分给各行 —— 只有码变."""
    return reorder_menu(ex, list(range(len(ex.options))), rng.sample(range(N_SLOTS), len(ex.options)))


def random_rows(ex: MenuExample, n: int, rng: random.Random) -> list[int]:
    """短菜单的行: 正确那一行加 n-1 个随机的其它行, 按原顺序."""
    others = [r for r in range(len(ex.options)) if r != ex.gold_idx]
    return sorted(rng.sample(others, n - 1) + [ex.gold_idx])


def top_rows(scores: list[float], gold_idx: int, n: int) -> list[int]:
    """短菜单的行: 正确那一行加分数最高的 n-1 个其它行, 按原顺序. scores 是模型给这道题各行的概率,
    留下的就是模型自己最容易混淆的那些."""
    others = sorted((r for r in range(len(scores)) if r != gold_idx), key=lambda r: -scores[r])
    return sorted(others[: n - 1] + [gold_idx])


# 一致性配对 (TrainConfig.consistency): 同一道题在一个 batch 里出两份, 行序不同; 题带 partner_query 时第二份
# 还换成材料的另一种说法. 两份的菜单分布按描述对齐后应当相同, 训练时用 Jensen-Shannon 散度把它们拉到一起 (loss.consistency_js).


def partner(ex: MenuExample, rng: random.Random) -> MenuExample:
    """同一道题的另一种排法: 行重新随机打乱, 与 ex 的顺序一定不同 (两项菜单就是对调), 连续编号 D0, D1, ...
    ex 带 partner_query 时上下文换成它 (同一份材料的另一种说法), 第二份自己不再带.
    问句、选项说明、软标签都跟着原题走. 码要换成随机码的话, 由之后的 RandomCodes 给."""
    k = len(ex.options)
    if k < 2:
        raise ValueError(f"a {k}-option menu has no second row order")
    rows = list(range(k))
    while rows == list(range(k)):
        rows = rng.sample(range(k), k)
    out = reorder_menu(ex, rows)
    if ex.partner_query is not None:
        out = replace(out, query=ex.partner_query, partner_query=None)
    return out


def with_partners(examples: list[MenuExample], rng: random.Random) -> list[MenuExample]:
    """每道题后面紧跟它的 partner: [a0, a0', a1, a1', ...]. 训练按相邻两条 (2i, 2i+1) 配对."""
    return [x for ex in examples for x in (ex, partner(ex, rng))]


def row_alignment(a: MenuExample, b: MenuExample) -> list[int]:
    """a 菜单第 j 行的描述在 b 菜单的第几行. 按描述 (options 里的类 id) 对齐, 与码无关.
    两份必须是同一道题: 问句、选项集合相同, 上下文相同或 b 读的是 a 的另一种说法 (a.partner_query)."""
    same_text = b.query == a.query or (a.partner_query is not None and b.query == a.partner_query)
    same = (a.question, a.context_label, sorted(a.options)) == (b.question, b.context_label, sorted(b.options))
    if not (same and same_text) or len(set(a.options)) != len(a.options):
        raise ValueError("row_alignment needs two orders of the same question with distinct options")
    where = {c: j for j, c in enumerate(b.options)}
    return [where[c] for c in a.options]


# 一致性评估 (scoring.consistency_eval): 同一批题排成几种随机的样子, 看选中的描述变不变.


def random_arrangement(ex: MenuExample, rng: random.Random) -> MenuExample:
    """评估用的一种随机排法: 行随机打乱, 每行从 N_SLOTS 个码里随机挑一个 (互不相同), 二元题也一样.
    行号和码都与描述无关, 只看描述作答的模型在每种排法下给每条描述的概率都相同."""
    k = len(ex.options)
    return reorder_menu(ex, rng.sample(range(k), k), rng.sample(range(N_SLOTS), k))


def arrangements(examples: list[MenuExample], passes: int, rng: random.Random) -> list[list[MenuExample]]:
    """passes 份, 第 p 份是每道题的第 p 种随机排法 (random_arrangement), 题目顺序与 examples 相同."""
    return [[random_arrangement(ex, rng) for ex in examples] for _ in range(passes)]


@dataclass(frozen=True)
class LabeledSet:
    queries: list[str]
    labels: list[int]  # 索引进 names
    names: dict[int, str]  # 类 id -> 给模型看的名字
    context_label: str = "Customer message"
    questions: list[str] | None = None  # 与 queries 平行; None 表示没有逐条问句
    question_default: str | None = field(default=None)  # 没有逐条问句时统一用它
    qtype: str = "choice"

    def make_example(self, i: int, options: list[int], gold_idx: int) -> MenuExample:
        q = self.questions[i] if self.questions is not None else self.question_default
        return MenuExample(
            query=self.queries[i], options=options, gold_idx=gold_idx, label=self.labels[i],
            option_names=[self.names[c] for c in options], context_label=self.context_label, question=q,
            qtype=self.qtype,
        )

    def build_examples(
        self, classes: list[int], k_range: tuple[int, int], rng: random.Random, pool: list[int] | None = None,
    ) -> list[MenuExample]:
        """给 label 落在 classes 里的每条各组一个菜单. 用于固定的评估集.
        菜单干扰项从 pool 里抽 (默认就是 classes); 给 pool 可以把合成意图掺进留出类的菜单."""
        allowed = set(classes)
        menu_pool = classes if pool is None else pool
        out = []
        for i, lab in enumerate(self.labels):
            if lab not in allowed:
                continue
            opts, gi = compose_menu(lab, menu_pool, draw_k(k_range, rng), rng)
            out.append(self.make_example(i, opts, gi))
        return out

    def sample_examples(
        self, classes: list[int], k_range: tuple[int, int], n: int, rng: random.Random,
        pool: list[int] | None = None, k_log: bool = False,
    ) -> list[MenuExample]:
        """随机抽 n 条 label 落在 classes 内的, 各配一个现组的菜单. 用于训练批.
        pool / k_log 见 build_examples / draw_k."""
        allowed = set(classes)
        menu_pool = classes if pool is None else pool
        idx = [i for i, lab in enumerate(self.labels) if lab in allowed]
        out = []
        for i in rng.sample(idx, n):
            opts, gi = compose_menu(self.labels[i], menu_pool, draw_k(k_range, rng, k_log), rng)
            out.append(self.make_example(i, opts, gi))
        return out
