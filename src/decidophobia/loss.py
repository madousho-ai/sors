"""训练目标: 交叉熵到正确的 D 码, 分母有三种 (training_loss 的 kind).

menu       只在这条样本给出的 k 个槽上做 softmax. 与推理形状一致, 但菜单外的 D 从不进分母,
           超出训练菜单长度的 D 永远不吃梯度.
all-slots  分母是全部 256 个 D 槽. 菜单外的 D 每一步都被压低, 训的是「只说菜单上有的码」.
vocab      分母是整个词表 (151936 个 token). 前两种给全部 D 码的 logit 同时加一个常数时 loss 不变,
           没有任何东西把 D 码整体推到 15 万个普通 token 之上; 这一种每一步都压低普通 token,
           训的是「答题位置只说 D 码」. 普通 token 的输出行冻结, 压低它们要靠 LoRA 改 h、以及 D 行自己抬高.

目标都是菜单第 gold 位绑的那个 D 码 (slot_ids 里的 id).
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


def all_slot_cross_entropy(
    logits: torch.Tensor, slot_ids: torch.Tensor, gold: torch.Tensor, d_ids: list[int],
) -> torch.Tensor:
    """分母是全部 D 槽 (256 个), 不论菜单几项; 目标是菜单第 gold 位的那个 D.

    菜单外的 D 也在分母里, 于是每一步都会被压低 —— 训练的是「只说菜单上有的码」.
    词表里其余的 token 仍不进分母.
    """
    d = torch.tensor(d_ids, device=logits.device)
    target_id = slot_ids.gather(1, gold[:, None])  # (B, 1)
    hit = target_id == d[None, :]  # (B, n_d)
    if not bool(hit.any(1).all()):
        raise ValueError("gold slot is not one of d_ids")
    return F.cross_entropy(logits[:, d], hit.int().argmax(1))


def vocab_cross_entropy(
    logits: torch.Tensor, slot_ids: torch.Tensor, gold: torch.Tensor, reduction: str = "mean",
) -> torch.Tensor:
    """分母是整个词表; 目标是菜单第 gold 位绑的那个 D 的 token id. 普通 token 的梯度就是它的概率,
    概率越高压得越狠. reduction="none" 给逐题的值, 评估用."""
    target = slot_ids.gather(1, gold[:, None]).squeeze(1)
    if bool((target < 0).any()):
        raise ValueError("gold points at a padding slot")
    return F.cross_entropy(logits, target, reduction=reduction)


LOSSES = ("menu", "all-slots", "vocab")


def training_loss(
    kind: str, logits: torch.Tensor, slot_ids: torch.Tensor, gold: torch.Tensor, d_ids: list[int],
) -> torch.Tensor:
    """训练循环用的损失. menu = 分母只有菜单 k 个槽; all-slots = 全部 D 槽; vocab = 整个词表."""
    if kind == "menu":
        return slot_cross_entropy(logits, slot_ids, gold)
    if kind == "all-slots":
        return all_slot_cross_entropy(logits, slot_ids, gold, d_ids)
    if kind == "vocab":
        return vocab_cross_entropy(logits, slot_ids, gold)
    raise ValueError(f"unknown loss {kind!r}; expected one of {LOSSES}")


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
