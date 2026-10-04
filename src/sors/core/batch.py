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
    """
    if architecture != "slots":
        from sors.core.decision_batch import collate_decisions
        return collate_decisions(examples, tokenizer, d_ids, k_max, layout, max_length, type_marker,
                                 context_marker, architecture)
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


def token_groups(lengths: Sequence[int], budget: int, unit: int = 1) -> list[list[int]]:
    """按 token 预算把一批行分组, 每组前向时补齐到组里最长那条: 补齐后的 token 数 (行数 × 最长) 不超过 budget.
    超过预算的单个单元自己一组. 相邻 unit 行 (JS 配对的两种排法) 是一个单元, 按其中最长那条算, 不拆开.

    单元按长度从长到短排 (等长保持批里的顺序), 只在这个顺序上切段: 先取组数最少的切法, 再取其中补齐最少的.
    组数决定每步前向的次数 (CPU 发射开销与同步次数), 补齐决定白算的 token; 显存峰值由 budget 封顶.
    lengths 是 CPU 上的整数, 整个分组不读 GPU."""
    if budget < 1 or unit < 1 or len(lengths) % unit:
        raise ValueError("token_groups needs a positive budget and complete units of adjacent rows")
    widths = [max(lengths[i:i + unit]) for i in range(0, len(lengths), unit)]
    order = sorted(range(len(widths)), key=lambda u: -widths[u])
    n = len(order)
    best: list[tuple[int, int]] = [(0, 0)] * (n + 1)  # 从第 i 个单元起切到底: (组数, 补齐 token)
    cut = [n] * (n + 1)
    for i in range(n - 1, -1, -1):
        best[i] = (n + 1, 0)
        for j in range(i + 1, n + 1):
            padded = (j - i) * unit * widths[order[i]]
            if padded > budget and j > i + 1:
                break
            groups, tokens = best[j]
            if (groups + 1, tokens + padded) < best[i]:
                best[i], cut[i] = (groups + 1, tokens + padded), j
    out, i = [], 0
    while i < n:
        out.append([u * unit + r for u in order[i:cut[i]] for r in range(unit)])
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
