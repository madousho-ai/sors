"""把一批 MenuExample 变成张量. 左填充, 答案位置永远是最后一列."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from sors.core.menu import MenuExample, row_alignment
from sors.core.prompt import DEFAULT_LAYOUT, encode_prompts, prompt_pieces
from sors.core.tokens import CONTEXT_TOKENS


def fit(ids: list[int], max_length: int, bounds: tuple[int, int] | None = None) -> list[int]:
    """超过 max_length 时截短, 最后一个 token (答案位置) 永远保住.
    bounds 给出上下文起止标记的 id 时, 先截两个标记之间 state 的左边, 标记与问题整段不动;
    state 截空了还超长, 再从整条的左边截. 不给 bounds 就直接从整条的左边截 (旧行为, BoolQ 的 passage 在最前面)."""
    over = len(ids) - max_length
    if over > 0 and bounds is not None:
        a, b = ids.index(bounds[0]) + 1, ids.index(bounds[1])
        ids = ids[:a] + ids[a + min(over, b - a):]
    return ids[-max_length:]


def collate(
    examples: list[MenuExample],
    tokenizer,
    d_ids: list[int],
    k_max: int,
    layout: str = DEFAULT_LAYOUT,
    max_length: int = 512,
    type_marker: bool = False,
    context_marker: bool = False,
    architecture: str = "slots",
    option_branches: bool = True,
    traces: bool = False,
) -> dict[str, torch.Tensor]:
    """返回
      input_ids      (B, L)  左填充
      attention_mask (B, L)
      slot_ids       (B, k_max)  第 j 列是菜单第 j 项绑的那个 D 的 id (默认 <|Dj|>, 见 MenuExample.codes),
                                 超出该样本菜单长度的位置填 -1
      gold           (B,)        正确选项在菜单里的位置
      target         (B, k_max)  菜单各行的目标概率: 样本带软标签 (MenuExample.target) 就是它, 否则 gold 那格 1;
                                 超出菜单长度的位置 0
    提示照 core.prompt 的片段编码: 文字里写着的保留 token 名是普通文字. 超长的截法见 fit.
    option_branches=False (candidate): 不带逐选项完整分支 option_input_ids / option_attention_mask, 只给打包前向用.
    traces=True (candidate): 另给 trace_input_ids / trace_attention_mask (B, T) 左填充, 没有 trace 的行全 0, 给蒸馏老师用.
    """
    if architecture != "slots":
        from sors.core.decision_batch import collate_decisions
        return collate_decisions(examples, tokenizer, d_ids, k_max, layout, max_length, type_marker,
                                 context_marker, architecture, option_branches=option_branches, traces=traces)
    pieces = [sum(prompt_pieces(ex, layout, type_marker, context_marker), []) for ex in examples]
    pad = tokenizer.pad_token_id
    bounds = tuple(tokenizer.convert_tokens_to_ids(CONTEXT_TOKENS)) if context_marker else None
    encs = [fit(e, max_length, bounds) for e in encode_prompts(tokenizer, pieces)]
    L = max(len(e) for e in encs)
    input_ids = torch.full((len(encs), L), pad, dtype=torch.long)
    attn = torch.zeros((len(encs), L), dtype=torch.long)
    for i, e in enumerate(encs):
        input_ids[i, L - len(e) :] = torch.tensor(e)
        attn[i, L - len(e) :] = 1
    return {"input_ids": input_ids, "attention_mask": attn, **targets(examples, d_ids, k_max)}


def targets(examples, d_ids, k_max):
    """Labels and output coordinates shared by all encoders."""
    slot_ids = torch.full((len(examples), k_max), -1, dtype=torch.long)
    target = torch.zeros((len(examples), k_max), dtype=torch.float32)
    for i, ex in enumerate(examples):
        slot_ids[i, : len(ex.options)] = torch.tensor([d_ids[c] for c in ex.slot_codes])
        if ex.target is None:
            target[i, ex.gold_idx] = 1.0
        else:
            target[i, : len(ex.options)] = torch.tensor(ex.target, dtype=torch.float32)
    gold = torch.tensor([ex.gold_idx for ex in examples], dtype=torch.long)
    return {"slot_ids": slot_ids, "gold": gold, "target": target}


def length_groups(attention_mask: torch.Tensor, n: int) -> list[torch.Tensor]:
    """把一批的行按真实长度分成 n 组, 每组各自前向时只补齐到组里最长的那条, 不再补到全批最长.
    行按长度从长到短排 (等长的保持批里的顺序), 再尽量均分地切成 n 段; 行数不够 n 就每行一组.
    n == 1 时就是整批原样、行序不动 —— 与不分组时喂进模型的是同一个张量."""
    B = attention_mask.shape[0]
    if n <= 1:
        return [torch.arange(B, device=attention_mask.device)]
    order = torch.sort(attention_mask.sum(dim=1), descending=True, stable=True).indices
    return list(torch.tensor_split(order, min(n, B)))


def token_groups(lengths: Sequence[int], budget: int, unit: int | Sequence[int] = 1) -> list[list[int]]:
    """按 token 预算把一批行分组, 每组前向时补齐到组里最长那条: 补齐后的 token 数 (行数 × 最长) 不超过 budget.
    超过预算的单个单元自己一组. 相邻 unit 行 (JS 配对的两种排法) 是一个单元, 按其中最长那条算, 不拆开.
    unit 也可以是一串单元的行数 (按顺序铺满整批, 例如原题与改动题连同各自的配对共 4 行一个单元).

    单元按长度从长到短排 (等长保持批里的顺序), 只在这个顺序上切段: 先取组数最少的切法, 再取其中补齐最少的.
    组数决定每步前向的次数 (CPU 发射开销与同步次数), 补齐决定白算的 token; 显存峰值由 budget 封顶.
    lengths 是 CPU 上的整数, 整个分组不读 GPU."""
    sizes = [unit] * (len(lengths) // unit) if isinstance(unit, int) and unit >= 1 and not len(lengths) % unit \
        else (list(unit) if not isinstance(unit, int) else [])
    if budget < 1 or not sizes or min(sizes) < 1 or sum(sizes) != len(lengths):
        raise ValueError("token_groups needs a positive budget and complete units of adjacent rows")
    starts = [sum(sizes[:u]) for u in range(len(sizes))]
    widths = [max(lengths[a:a + k]) for a, k in zip(starts, sizes)]
    order = sorted(range(len(widths)), key=lambda u: -widths[u])
    n = len(order)
    best: list[tuple[int, int]] = [(0, 0)] * (n + 1)  # 从第 i 个单元起切到底: (组数, 补齐 token)
    cut = [n] * (n + 1)
    for i in range(n - 1, -1, -1):
        best[i] = (n + 1, 0)
        for j in range(i + 1, n + 1):
            padded = sum(sizes[u] for u in order[i:j]) * widths[order[i]]
            if padded > budget and j > i + 1:
                break
            groups, tokens = best[j]
            if (groups + 1, tokens + padded) < best[i]:
                best[i], cut[i] = (groups + 1, tokens + padded), j
    out, i = [], 0
    while i < n:
        out.append([starts[u] + r for u in order[i:cut[i]] for r in range(sizes[u])])
        i = cut[i]
    return out


def trim_left_padding(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """去掉每一行都是填充的那些左侧列. 左填充下答案位置仍是最后一列."""
    keep = int(attention_mask.sum(dim=1).max())
    return input_ids[:, -keep:], attention_mask[:, -keep:]


def pair_alignment(examples: list[MenuExample], k_max: int) -> torch.Tensor:
    """(n, k_max), n = len(examples) // 2. 相邻两条 (2i, 2i+1) 是同一道题的两种排法 (menu.with_partners);
    第 i 行第 j 列 = 第 2i 条菜单第 j 行的描述在第 2i+1 条菜单的第几行 (menu.row_alignment). 菜单之外补 -1."""
    if len(examples) % 2:
        raise ValueError(f"{len(examples)} examples cannot be split into pairs")
    out = torch.full((len(examples) // 2, k_max), -1, dtype=torch.long)
    for i in range(0, len(examples), 2):
        rows = row_alignment(examples[i], examples[i + 1])
        out[i // 2, : len(rows)] = torch.tensor(rows)
    return out


# 原题与改动题 (MenuExample.contrast_group) 的成对损失: 配对在批里的位置与两种答案在各行菜单上的位置.


def group_contrasts(examples: list[MenuExample]) -> list[MenuExample]:
    """同一个 contrast_group 的题挪到一起, 放在其中最早那道的位置; 其余题顺序不动, 不消耗随机数."""
    out, done = [], set()
    for ex in examples:
        g = ex.contrast_group
        if g is None:
            out.append(ex)
        elif g not in done:
            done.add(g)
            out += [e for e in examples if e.contrast_group == g]
    return out


def contrast_units(examples: list[MenuExample], width: int) -> list[int]:
    """按 token 预算分组用的单元行数: 每 width 行 (一道题和它的一致性配对) 一个单元, 同一 contrast_group
    相邻的几个单元并成一个, 成对损失需要两道题在同一次前向里."""
    sizes, last = [], None
    for i in range(0, len(examples), width):
        g = examples[i].contrast_group
        if g is not None and g == last:
            sizes[-1] += width
        else:
            sizes.append(width)
        last = g
    return sizes


def contrast_rows(examples: list[MenuExample], qtypes=None) -> dict[str, torch.Tensor] | None:
    """成对损失要的下标. 一组 = contrast_group 相同的行 (原题、改动题和各自的一致性配对), 正好有两种硬标签答案
    A、B (按第一次出现排), 而且每一行的菜单上两者都在; 不满足的组跳过. 没有可用的组时返回 None.
    qtypes 给了就只要组里每一行的题型 (MenuExample.qtype) 都在其中的组, 例如只对是非题算成对损失.
    返回 row (批里的行) / group (第几组, 0 起) / is_b (这一行答 B) / pos_a / pos_b (A、B 在这一行菜单上的位置)."""
    groups: dict[int, list[int]] = {}
    for i, ex in enumerate(examples):
        if ex.contrast_group is not None:
            groups.setdefault(ex.contrast_group, []).append(i)
    rows, gid, is_b, pos_a, pos_b = [], [], [], [], []
    for members in groups.values():
        labels = list(dict.fromkeys(examples[i].label for i in members))
        if len(labels) != 2 or any(examples[i].target is not None for i in members):
            continue
        if qtypes is not None and any(examples[i].qtype not in qtypes for i in members):
            continue
        a, b = labels
        if not all(a in examples[i].options and b in examples[i].options for i in members):
            continue
        g = len(set(gid))
        for i in members:
            rows.append(i); gid.append(g); is_b.append(examples[i].label == b)
            pos_a.append(examples[i].options.index(a)); pos_b.append(examples[i].options.index(b))
    if not rows:
        return None
    t = lambda x, dt=torch.long: torch.tensor(x, dtype=dt)  # noqa: E731
    return {"row": t(rows), "group": t(gid), "is_b": t(is_b, torch.bool), "pos_a": t(pos_a), "pos_b": t(pos_b)}


def score_levels(examples: list[MenuExample], k_max: int) -> torch.Tensor:
    """(n, k_max): 评分题 (qtype score) 菜单第 j 行的级别 (类 id, 0 最低); 其余题、菜单之外补 -1.
    行序打乱、换码都不影响级别, 级别跟着类 id 走."""
    out = torch.full((len(examples), k_max), -1, dtype=torch.long)
    for i, ex in enumerate(examples):
        if ex.qtype == "score":
            out[i, : len(ex.options)] = torch.tensor(ex.options)
    return out
