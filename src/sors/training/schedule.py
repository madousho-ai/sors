"""学习率日程. 返回乘在基础 lr 上的倍率, 交给 LambdaLR.

上一轮恒定 lr 的 unseen 曲线在评估之间抖 ±3 点 (0.90 -> 0.86 -> 0.90 -> 0.87),
大于 n=680 的噪声 (SE ~1.3), 单批 loss 在 0.005 与 0.35 之间跳 —— 后半程步子太大.
"""

from __future__ import annotations

import math


def lr_scale(step: int, warmup: int, total: int, kind: str = "cosine") -> float:
    """kind='constant' 恒为 1. kind='cosine': 0..warmup 线性升到 1, 之后余弦降到 total 处为 0."""
    if kind == "constant":
        return 1.0
    if warmup > 0 and step < warmup:
        return step / warmup
    if total <= warmup:
        return 1.0
    progress = min(1.0, (step - warmup) / (total - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * progress))
