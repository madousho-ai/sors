"""视觉输入: 冻结的 Qwen3.5 视觉编码器把图片编成向量, 写到文字序列里图片占位符 (<|image_pad|>) 的位置上.

底座的 deepstack_visual_indexes 为空, 视觉特征只在嵌入层进来, 之后文字主干的每一层照常算.
图片在 state 里, 选项分支只有文字.

位置是 M-RoPE 的三个坐标 (时间, 行, 列). 文字 token 三个坐标相同, 每个 +1; 一张图的占位符排在
合并后的网格上 (时间轴都是图开始的位置, 行列各自从那里数起), 整张图走完位置只前进 max(行数, 列数) 格.
所以带图的前缀, 它的 token 数与它之后的起始位置不再相等; 分支要从 rope_positions 给出的位置接着数.
"""

from __future__ import annotations

import base64
import binascii
import io
import re
from dataclasses import dataclass

import torch

SPATIAL_MERGE = 2  # 视觉编码器把 2x2 个 patch 合成一个占位符 (vision_config.spatial_merge_size)
VISION_TOKENS = ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]  # Qwen3.5 tokenizer 自带
# state 文字里一张图的位置. 私用区字符, 调用方的文字里出现就拒收 (split_data_uris), 伪造不出图片位置
IMAGE_MARK = "\ue000"
_DATA_URI = re.compile(r"data:image/(?:png|jpeg|jpg|gif|webp);base64,([A-Za-z0-9+/]+={0,2})")


@dataclass(frozen=True, eq=False)
class VisionImage:
    """一张图预处理后的样子: patch 网格 (时间, 高, 宽) 与展平的像素块 (时间 x 高 x 宽, 通道 x patch 面积)."""
    grid: tuple[int, int, int]
    pixels: torch.Tensor

    @property
    def tokens(self) -> int:
        """它在提示里占几个 <|image_pad|>."""
        t, h, w = self.grid
        return t * (h // SPATIAL_MERGE) * (w // SPATIAL_MERGE)


def split_data_uris(text: str) -> tuple[str, list[bytes]]:
    """把 text 里的 data:image/...;base64,... 换成 IMAGE_MARK, 依次返回图片的原始字节.

    周围的文字原样保留 (JSON state 里的引号也在). 解不开的 base64 或不是图片的内容报错;
    text 里本来就有 IMAGE_MARK 也报错.
    """
    from PIL import Image, UnidentifiedImageError

    if IMAGE_MARK in text:
        raise ValueError("the state contains the reserved image mark U+E000")
    images = []

    def take(match):
        try:
            data = base64.b64decode(match.group(1), validate=True)
            Image.open(io.BytesIO(data)).verify()
        except (binascii.Error, UnidentifiedImageError, OSError, SyntaxError) as exc:
            raise ValueError(f"cannot decode the image data URI at offset {match.start()}: {exc}") from exc
        images.append(data)
        return IMAGE_MARK

    return _DATA_URI.sub(take, text), images


def image_batch(examples) -> dict[str, torch.Tensor]:
    """批里各行的 image_grids (B, M, 3) 与 image_pixels (B, P, C): 各行的图依次排, 不足处补零."""
    grids = torch.zeros(len(examples), max(len(e.images) for e in examples), 3, dtype=torch.long)
    width = max(sum(len(i.pixels) for i in e.images) for e in examples)
    first = next(i.pixels for e in examples for i in e.images)
    pixels = first.new_zeros(len(examples), width, first.shape[-1])
    for r, ex in enumerate(examples):
        if ex.images:
            grids[r, :len(ex.images)] = torch.tensor([i.grid for i in ex.images])
            rows = torch.cat([i.pixels for i in ex.images])
            pixels[r, :len(rows)] = rows
    return {"image_grids": grids, "image_pixels": pixels}


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


def row_grids(grids: torch.Tensor) -> list[tuple[int, int, int]]:
    """批里一行的 image_grids (M, 3) 去掉全零的空位."""
    return [tuple(g) for g in grids.tolist() if any(g)]


def has_images(batch) -> bool:
    return "image_embeds" in batch or "image_pixels" in batch


def preprocess_images(processor, images: list[bytes]) -> list[VisionImage]:
    """图片字节 -> VisionImage, 每张图单独过预处理器 (各自的分辨率与网格). 透明通道与调色板都按 RGB 读."""
    from PIL import Image

    out = []
    for data in images:
        enc = processor(images=[Image.open(io.BytesIO(data)).convert("RGB")], return_tensors="pt")
        out.append(VisionImage(grid=tuple(int(x) for x in enc["image_grid_thw"][0]), pixels=enc["pixel_values"]))
    return out


def load_processor(model_id, *, max_pixels: int | None = None, revision=None, local_files_only=False):
    """底座的图像预处理器. max_pixels 给了就把单张图缩到不超过这么多像素 (一个占位符是 32x32 像素);
    Qwen3.5 自带的上限是 4096x4096, 一张图就有 16384 个占位符."""
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(model_id, revision=revision, local_files_only=local_files_only)
    if max_pixels is not None:
        low = min(processor.size["shortest_edge"], max_pixels)
        processor.size = {"shortest_edge": low, "longest_edge": max_pixels}
    return processor


def attach_vision(model, visual, image_token_id: int) -> None:
    """把冻结的视觉编码器挂到决策模型上 (model.visual), 带 image_pixels 的批在前向时由它编码."""
    visual.requires_grad_(False)
    model.visual = visual.eval()
    model.image_token_id = image_token_id


@torch.no_grad()
def encode_images(visual, pixels: torch.Tensor, grids: torch.Tensor) -> torch.Tensor:
    """批里的 image_pixels (B, P, C) 与 image_grids (B, M, 3) -> image_embeds (B, N, hidden), 不足处补零.
    全批的图一次过编码器; 编码器内部的注意力按图分段, 结果与逐张编码相同."""
    rows = [row_grids(g) for g in grids]
    sizes = [sum(t * h * w for t, h, w in r) for r in rows]
    tokens = [sum(t * (h // SPATIAL_MERGE) * (w // SPATIAL_MERGE) for t, h, w in r) for r in rows]
    dtype = next(visual.parameters()).dtype
    flat = torch.cat([pixels[i, :n] for i, n in enumerate(sizes)]).to(dtype)
    grid = torch.tensor([g for r in rows for g in r], device=pixels.device)
    encoded = visual(flat, grid_thw=grid).pooler_output
    out = encoded.new_zeros(len(rows), max(tokens), encoded.shape[-1])
    for i, part in enumerate(encoded.split(tokens)):
        out[i, :len(part)] = part
    return out


def load_vision(model_id, *, dtype=torch.bfloat16, device="cuda", revision=None, local_files_only=False):
    """从多模态底座 (Qwen3.5) 的权重里只取视觉编码器 (model.visual.*), 冻结. 返回 (编码器, 图片占位符 id)."""
    from pathlib import Path

    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    config = AutoConfig.from_pretrained(model_id, revision=revision, local_files_only=local_files_only)
    if getattr(config, "vision_config", None) is None:
        raise ValueError(f"{model_id} has no vision encoder")
    directory = Path(model_id)
    if not directory.is_dir():
        from huggingface_hub import snapshot_download
        directory = Path(snapshot_download(model_id, revision=revision, local_files_only=local_files_only,
                                           allow_patterns=["*.safetensors", "*.json"]))
    prefix = "model.visual."
    state = {}
    for path in sorted(directory.glob("*.safetensors")):
        with safe_open(path, framework="pt") as f:
            state.update({k[len(prefix):]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)})  # noqa: SIM118 (safe_open 不是 dict)
    visual = Qwen3_5VisionModel(config.vision_config)
    visual.load_state_dict(state, strict=True)
    visual.to(device=device, dtype=dtype).requires_grad_(False)
    return visual.eval(), config.image_token_id


def image_inputs(embed, ids, mask, image_token_id: int, embeds, grids):
    """带图的一批左填充序列 -> (inputs_embeds (B, L, hidden), position_ids (4, B, L)).

    embed: 主干的嵌入层 (连同 D 行替换). embeds (B, P, hidden): 各行依次出现的图像向量, 多出的是填充.
    grids (B, M, 3): 各行每张图的网格, 全零行是空位. position_ids 第 0 行是序列序号 (主干拿它建注意力掩码),
    后三行是 M-RoPE 坐标; 填充位都是 0.
    """
    x = embed(ids).clone()
    text = (mask.long().cumsum(-1) - 1).clamp_min(0)
    pos = text[None].repeat(4, 1, 1)
    for r in range(ids.shape[0]):
        valid = mask[r].bool()
        p, _ = rope_positions(ids[r][valid], row_grids(grids[r]), image_token_id)
        pos[1:, r, valid] = p.to(pos.device)
        spots = valid & (ids[r] == image_token_id)
        n = int(spots.sum())
        if n > embeds.shape[1]:
            raise ValueError(f"row {r} has {n} image placeholders but only {embeds.shape[1]} image vectors")
        x[r, spots] = embeds[r, :n].to(x.dtype)
    return x, pos
