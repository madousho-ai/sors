"""训练目标: 只在这条样本给出的 k 个槽上做 softmax, 交叉熵到正确槽.

与推理形状一致 —— 调用方给几个选项就在几个上归一. 词表里其余 15 万个 token
不进分母: 这里训的是「选哪个」, 「要不要说 D-token 而非别的」由嵌入行的更新顺带解决.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gather_slot_logits(logits: torch.Tensor, slot_ids: torch.Tensor) -> torch.Tensor:
    """(B, V) x (B, K) -> (B, K); slot_ids == -1 的位置填 -inf."""
    mask = slot_ids >= 0
    got = logits.gather(1, slot_ids.clamp_min(0))
    return got.masked_fill(~mask, float("-inf"))


def slot_cross_entropy(logits: torch.Tensor, slot_ids: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(gather_slot_logits(logits, slot_ids), gold)


def answer_mass(
    logits: torch.Tensor, slot_ids: torch.Tensor, d_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """格式遵从的三个读数, 都在全词表 softmax 下量. 训练目标只在 k 个槽上归一, 管不到这里.

    m_answer   这条样本菜单里 k 个槽的概率之和 —— 与 baseline 脚本的 m_answer 同一个量
    m_offmenu  其余 D 槽的概率之和: 说了 D-token 但指向菜单里没有的位置
    top1_in    全词表 argmax 是不是菜单里的槽, 即自由贪心解码会不会吐出合法答案
    """
    p = torch.softmax(logits.float(), dim=-1)
    m = (p.gather(1, slot_ids.clamp_min(0)) * (slot_ids >= 0)).sum(1)
    all_d = p[:, torch.tensor(d_ids, device=logits.device)].sum(1)
    top1_in = (logits.argmax(-1, keepdim=True) == slot_ids).any(1)
    return m, all_d - m, top1_in
