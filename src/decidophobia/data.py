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

from decidophobia.tokens import N_SLOTS


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

    def __post_init__(self):
        # D 槽只有 N_SLOTS 个. 把菜单压到这个数以内是各数据集管线的责任, 压不住就在抽样当下报错.
        if len(self.options) > N_SLOTS:
            raise ValueError(f"menu has {len(self.options)} options, only {N_SLOTS} D slots; "
                             "the dataset pipeline must cap it")
        if self.codes is not None:
            c = self.codes
            if len(c) != len(self.options) or len(set(c)) != len(c) or not all(0 <= x < N_SLOTS for x in c):
                raise ValueError(f"codes must be {len(self.options)} distinct slots in 0..{N_SLOTS - 1}, got {c}")

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
    """训练抽题的最后一步: 每条选择题 (qtype choice) 以概率 rate 换一套码, 其余保持 D0, D1, ... 连续编号;
    二元题 (BoolQ) 永远连续编号. 菜单第 i 项写成这套里的第 i 个, 答案是正确那一项旁边写的码, 与它排第几行无关.

    设计目的是光看编号推不出答案: 一个码出现在 k 项菜单上时, 它是答案的概率必须是 1/k, 对哪个码都一样.
    所以换码时不单独给答案挑码, 而是整套一起挑: 取到目前为止上菜单次数最少的 k 个码 (并列时随机),
    再随机排进菜单 —— 答案落在这套里的哪一个完全随机. 均衡的是上菜单的次数, 当答案的次数随之期望相同.
    计数覆盖所有选择题, 连续编号那部分也算 (它们只占 D0..D(k-1)), 跨调用保留 —— 一个 run 用一个实例.
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
            if ex.qtype == "choice":
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
    codes 给新菜单每一行绑的码, None = 连续编号 D0, D1, ..."""
    if len(set(rows)) != len(rows) or not all(0 <= r < len(ex.options) for r in rows):
        raise ValueError(f"rows must be distinct rows of a {len(ex.options)}-option menu, got {rows}")
    if ex.gold_idx not in rows:
        raise ValueError(f"the gold row {ex.gold_idx} must stay on the menu")
    return replace(ex, options=[ex.options[r] for r in rows], option_names=[ex.option_names[r] for r in rows],
                   gold_idx=rows.index(ex.gold_idx), codes=codes)


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

