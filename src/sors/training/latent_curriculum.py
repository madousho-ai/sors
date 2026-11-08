"""隐式推理的课程 (Coconut 式): 训练时把带计算过程的题 (MenuExample.cot) 一句一句地换成隐式位置.

计算过程按句切成步 (split_steps). 第 s 阶段 (latent_stage):
  - 带计算过程的题: 前 min(s, 句数) 句换成同样多个隐式位置, 剩下的句子仍是文字, 接在隐式位置后面,
    训练时对它们算下一个词的损失 (core.latent); 选项分支接在文字后面.
  - 最后阶段 (s = k_max): 全部换成 k_max 个隐式位置, 多出的句子不再出现 (并进最后一步).
  - 不带计算过程的题: 一直用 k_max 个隐式位置 (latent_n 留 None), 与推理时一样.
推理时不写文字, 每题都是 k_max 个隐式位置.
"""

from __future__ import annotations

import re
from dataclasses import replace
from fractions import Fraction

_END = re.compile(r"[.?!][\"'”’)\]]*(?=\s|$)|\n")  # 句末: 句号 / 问号 / 叹号 (可带收尾的引号括号), 或换行


def split_steps(cot: str | None) -> list[str]:
    """计算过程按句切开 (一句一步); 空白句丢掉."""
    if not cot:
        return []
    out, start = [], 0
    for m in _END.finditer(cot):
        piece = cot[start:m.end()].strip()
        if piece:
            out.append(piece)
        start = m.end()
    rest = cot[start:].strip()
    if rest:
        out.append(rest)
    return out


def _step_ends(cot: str) -> list[int]:
    """split_steps 每一步在原文里结束的字符位置, 用来取剩下的原文 (保留原来的空白与换行)."""
    ends, start = [], 0
    for m in _END.finditer(cot):
        if cot[start:m.end()].strip():
            ends.append(m.end())
        start = m.end()
    if cot[start:].strip():
        ends.append(len(cot))
    return ends


def latent_stage(step: int, total: int, k_max: int, schedule: tuple[float, float]) -> int:
    """第 step 步 (共 total 步) 所在的阶段: a 之前是 0 (计算过程全是文字), 从 a 起每 (b - a) / k_max 升一级,
    到 k_max 为止 (a + (b - a)(k_max - 1) / k_max 之后一直是最后阶段). 比例用分数算, 边界不受浮点误差影响."""
    a, b = (Fraction(str(x)) for x in schedule)
    t = Fraction(step, total)
    if t < a:
        return 0
    return min(k_max, int((t - a) / (b - a) * k_max) + 1)


def curriculum_examples(examples, stage: int, k_max: int):
    """按阶段改写一批题: 带 cot 的题设 latent_n / latent_text, 其余原样."""
    out = []
    for ex in examples:
        if not ex.cot:
            out.append(ex)
            continue
        if stage >= k_max:
            out.append(replace(ex, latent_n=k_max, latent_text=None))
            continue
        ends = _step_ends(ex.cot)
        n = min(stage, len(ends))
        rest = ex.cot[ends[n - 1]:].strip() if n else ex.cot.strip()
        out.append(replace(ex, latent_n=n, latent_text=rest or None))
    return out
