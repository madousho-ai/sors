"""视觉输入: 冻结的 Qwen3.5 视觉编码器把图片编成向量, 写到文字序列里图片占位符 (<|image_pad|>) 的位置上.

底座的 deepstack_visual_indexes 为空, 视觉特征只在嵌入层进来, 之后文字主干的每一层照常算.
图片在 state 里, 选项分支只有文字.

位置是 M-RoPE 的三个坐标 (时间, 行, 列). 文字 token 三个坐标相同, 每个 +1; 一张图的占位符排在
合并后的网格上 (时间轴都是图开始的位置, 行列各自从那里数起), 整张图走完位置只前进 max(行数, 列数) 格.
所以带图的前缀, 它的 token 数与它之后的起始位置不再相等; 分支要从 rope_positions 给出的位置接着数.
"""

from __future__ import annotations

import torch

SPATIAL_MERGE = 2  # 视觉编码器把 2x2 个 patch 合成一个占位符 (vision_config.spatial_merge_size)


def rope_positions(ids, grids, image_token_id: int, start: int = 0,
                   merge: int = SPATIAL_MERGE) -> tuple[torch.Tensor, int]:
    """ids 的 M-RoPE 位置 (3, len(ids)), 以及其后下一个 token 的位置.

    grids: 依出现顺序每张图的 (时间, 高, 宽) patch 网格 (图像预处理器的 image_grid_thw).
    一张图在 ids 里占 时间 x (高/merge) x (宽/merge) 个连续的占位符; 数目对不上就报错.
    """
    ids = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)
    pieces, pos, i, pending = [], start, 0, list(grids)
    while i < len(ids):
        if ids[i] != image_token_id:
            j = i
            while j < len(ids) and ids[j] != image_token_id:
                j += 1
            pieces.append(torch.arange(pos, pos + j - i).expand(3, -1))
            pos += j - i
            i = j
            continue
        if not pending:
            raise ValueError("more image placeholders than images")
        t, h, w = (int(x) for x in pending.pop(0))
        rows, cols = h // merge, w // merge
        n = t * rows * cols
        if ids[i:i + n] != [image_token_id] * n:
            raise ValueError(f"an image with grid {(t, h, w)} needs {n} consecutive placeholders")
        tt, hh, ww = torch.meshgrid(torch.arange(t), torch.arange(rows), torch.arange(cols), indexing="ij")
        pieces.append(torch.stack([tt + pos, hh + pos, ww + pos]).reshape(3, -1))
        pos += max(rows, cols)
        i += n
    if pending:
        raise ValueError(f"{len(pending)} images have no placeholders")
    out = torch.cat(pieces, dim=1) if pieces else torch.zeros(3, 0, dtype=torch.long)
    return out, pos
