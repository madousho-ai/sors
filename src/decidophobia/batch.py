"""把一批 MenuExample 变成张量. 左填充, 答案位置永远是最后一列."""

from __future__ import annotations

import torch

from decidophobia.data import MenuExample
from decidophobia.prompt import DEFAULT_LAYOUT, render_menu


def collate(
    examples: list[MenuExample],
    tokenizer,
    d_ids: list[int],
    k_max: int,
    layout: str = DEFAULT_LAYOUT,
    max_length: int = 512,
    type_marker: bool = False,
) -> dict[str, torch.Tensor]:
    """返回
      input_ids      (B, L)  左填充
      attention_mask (B, L)
      slot_ids       (B, k_max)  第 j 列是菜单第 j 项绑的那个 D 的 id (默认 <|Dj|>, 见 MenuExample.codes),
                                 超出该样本菜单长度的位置填 -1
      gold           (B,)        正确选项在菜单里的位置
    """
    texts = [render_menu(ex, layout, type_marker) for ex in examples]
    pad = tokenizer.pad_token_id
    # 超长的从左边截 (BoolQ 的 passage 在最前面), 答案位置永远保住
    encs = [tokenizer.encode(t, add_special_tokens=False)[-max_length:] for t in texts]
    L = max(len(e) for e in encs)
    input_ids = torch.full((len(encs), L), pad, dtype=torch.long)
    attn = torch.zeros((len(encs), L), dtype=torch.long)
    for i, e in enumerate(encs):
        input_ids[i, L - len(e) :] = torch.tensor(e)
        attn[i, L - len(e) :] = 1
    slot_ids = torch.full((len(examples), k_max), -1, dtype=torch.long)
    for i, ex in enumerate(examples):
        slot_ids[i, : len(ex.options)] = torch.tensor([d_ids[c] for c in ex.slot_codes])
    gold = torch.tensor([ex.gold_idx for ex in examples], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attn, "slot_ids": slot_ids, "gold": gold}
