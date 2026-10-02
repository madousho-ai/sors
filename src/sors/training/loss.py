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

import math

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


def smooth_target(target: torch.Tensor, slot_ids: torch.Tensor, eps: float) -> torch.Tensor:
    """标签平滑, 只在菜单的 k 行上摊: (1 - eps) * target + eps / k. 补位列 (slot_ids == -1) 仍是 0.
    硬标签的正确那一行因此是 1 - eps + eps / k, 其余每行 eps / k. eps 0 原样返回同一个张量."""
    if eps == 0:
        return target
    on = (slot_ids >= 0).to(target.dtype)
    return (1 - eps) * target + eps * on / on.sum(1, keepdim=True)


def menu_log_probs(
    kind: str, logits: torch.Tensor, slot_ids: torch.Tensor, d_ids: list[int],
) -> torch.Tensor:
    """(B, K): 菜单每一行那个 D 码的对数概率, 分母按 kind 取 (菜单 k 个槽 / 全部 D 槽 / 整个词表). 补位列是 -inf."""
    slot = gather_slot_logits(logits, slot_ids)
    if kind == "menu":
        z = torch.logsumexp(slot, 1, keepdim=True)
    elif kind == "all-slots":
        d = torch.tensor(d_ids, device=logits.device)
        if not bool((torch.isin(slot_ids, d) | (slot_ids < 0)).all()):
            raise ValueError("a menu slot is not one of d_ids")
        z = torch.logsumexp(logits[:, d], 1, keepdim=True)
    elif kind == "vocab":
        z = torch.logsumexp(logits, 1, keepdim=True)
    else:
        raise ValueError(f"unknown loss {kind!r}; expected one of {LOSSES}")
    return slot - z


def target_cross_entropy(
    kind: str, logits: torch.Tensor, slot_ids: torch.Tensor, target: torch.Tensor, d_ids: list[int],
) -> torch.Tensor:
    """对目标分布的交叉熵 -Σ target · log p, 批内取平均. target (B, K) 与 slot_ids 平行;
    one-hot 目标时等于按 gold 下标算的那一种."""
    if bool(((target > 0) & (slot_ids < 0)).any()):
        raise ValueError("target puts probability on a padding slot")
    logp = menu_log_probs(kind, logits, slot_ids, d_ids)
    return -torch.where(target > 0, target * logp, torch.zeros_like(logp)).sum(1).mean()


def consistency_js(logits: torch.Tensor, slot_ids: torch.Tensor, align: torch.Tensor) -> torch.Tensor:
    """同一道题两种排法之间的 Jensen-Shannon 散度, 按对取平均. 行 2i 与 2i+1 是第 i 对,
    align (n, K) 是 batch.pair_alignment 给的对齐: 前一份第 j 行的描述在后一份的第 align[i, j] 行, 补位 -1.

    两份各自只在菜单 k 行上 softmax (不论训练 loss 用哪个分母: 这里只比概率在菜单上怎么分),
    按描述对齐成 P、Q, M = (P + Q) / 2, JS = ½ KL(P‖M) + ½ KL(Q‖M). 0 = 两种排法给每条描述的概率相同,
    上限 ln 2. 梯度同时流进两份, 把它们往 M 拉; 不需要标签."""
    if logits.shape[0] != 2 * align.shape[0]:
        raise ValueError(f"{logits.shape[0]} rows of logits for {align.shape[0]} pairs")
    on = align >= 0
    counts = (slot_ids >= 0).sum(1).reshape(-1, 2)
    if not bool((counts == on.sum(1)[:, None]).all()):
        raise ValueError("align does not cover each menu's rows")
    # 补位先填 0 再做运算: -inf 减 -inf 是 nan, 即使事后被 where 丢掉, 反传时也会把 nan 带进梯度
    la = torch.log_softmax(gather_slot_logits(logits[0::2], slot_ids[0::2]), 1).masked_fill(~on, 0.0)
    lb = torch.log_softmax(gather_slot_logits(logits[1::2], slot_ids[1::2]), 1)
    lb = lb.gather(1, align.clamp_min(0)).masked_fill(~on, 0.0)
    lm = torch.logaddexp(la, lb) - math.log(2)
    term = 0.5 * (la.exp() * (la - lm) + lb.exp() * (lb - lm))
    return torch.where(on, term, torch.zeros_like(term)).sum(1).mean()


def menu_hit_counts(
    logits: torch.Tensor, slot_ids: torch.Tensor, target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """训练批上的正确率读数, 返回 (答对几道, 算了几道). 模型的选择是菜单 k 行里 logit 最高的那一行
    (菜单外的 D 码、普通 token 不参与, 与评估时的 accuracy 同一种读法); 目标 target (B, K) 是 collate 给的分布,
    平滑之前. 选中的那一行在 target 上是最大值 (并列最大时选中其中任意一行都算) 就是答对.
    target 在菜单上处处相等的题 (文本没提、标成均匀的那些) 没有答案, 不计入分母."""
    on = slot_ids >= 0
    pick = gather_slot_logits(logits, slot_ids).argmax(1, keepdim=True)
    top = target.masked_fill(~on, float("-inf")).amax(1)
    low = target.masked_fill(~on, float("inf")).amin(1)
    scored = top - low > 1e-6
    hit = (target.gather(1, pick).squeeze(1) >= top - 1e-6) & scored
    return hit.sum(), scored.sum()


def menu_hits(logits: torch.Tensor, slot_ids: torch.Tensor, target: torch.Tensor) -> tuple[int, int]:
    """Host-facing accuracy counts; training batches the device counts with its other metrics."""
    hits, scored = menu_hit_counts(logits, slot_ids, target)
    return int(hits), int(scored)


def training_loss(
    kind: str, logits: torch.Tensor, slot_ids: torch.Tensor, gold: torch.Tensor, d_ids: list[int],
    target: torch.Tensor | None = None,
) -> torch.Tensor:
    """训练循环用的损失. menu = 分母只有菜单 k 个槽; all-slots = 全部 D 槽; vocab = 整个词表.
    target 给了就对这个分布求交叉熵 (软标签、标签平滑); 不给就只认 gold, 与加 target 之前逐位相同."""
    if target is not None:
        return target_cross_entropy(kind, logits, slot_ids, target, d_ids)
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
