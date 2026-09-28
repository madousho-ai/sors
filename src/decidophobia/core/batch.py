"""把一批 MenuExample 变成张量. 左填充, 答案位置永远是最后一列."""

from __future__ import annotations

import torch

from decidophobia.core.menu import MenuExample, row_alignment
from decidophobia.core.prompt import DEFAULT_LAYOUT, encode_prompts, prompt_pieces
from decidophobia.core.tokens import CONTEXT_TOKENS


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
    slot_ids = torch.full((len(examples), k_max), -1, dtype=torch.long)
    target = torch.zeros((len(examples), k_max), dtype=torch.float32)
    for i, ex in enumerate(examples):
        slot_ids[i, : len(ex.options)] = torch.tensor([d_ids[c] for c in ex.slot_codes])
        if ex.target is None:
            target[i, ex.gold_idx] = 1.0
        else:
            target[i, : len(ex.options)] = torch.tensor(ex.target, dtype=torch.float32)
    gold = torch.tensor([ex.gold_idx for ex in examples], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attn, "slot_ids": slot_ids, "gold": gold, "target": target}


def length_groups(attention_mask: torch.Tensor, n: int) -> list[torch.Tensor]:
    """把一批的行按真实长度分成 n 组, 每组各自前向时只补齐到组里最长的那条, 不再补到全批最长.
    行按长度从长到短排 (等长的保持批里的顺序), 再尽量均分地切成 n 段; 行数不够 n 就每行一组.
    n == 1 时就是整批原样、行序不动 —— 与不分组时喂进模型的是同一个张量."""
    B = attention_mask.shape[0]
    if n <= 1:
        return [torch.arange(B, device=attention_mask.device)]
    order = torch.sort(attention_mask.sum(dim=1), descending=True, stable=True).indices
    return list(torch.tensor_split(order, min(n, B)))


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
