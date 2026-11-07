"""冻结的视觉编码器: 图片字节 -> VisionImage (预处理), 批里的像素 -> 图像向量 (前向时), 从底座权重载入 (CPU).

用一层的随机 Qwen3.5 视觉编码器, 输出宽度与两层的随机文字主干 (hidden 64) 相同.
"""

import io
import tempfile
from pathlib import Path

import pytest
import torch
from test_candidate import candidate_batch
from test_vision_candidate import reference_logits
from test_vision_prompt import fake_image, png, vision_tokenizer

from sors.core.vision import IMAGE_MARK, VISION_TOKENS, attach_vision, preprocess_images


def tiny_vision_config():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    return Qwen3_5VisionConfig(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=64,
                               patch_size=16, temporal_patch_size=2, spatial_merge_size=2, num_position_embeddings=16)


def tiny_visual(seed=0):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    torch.manual_seed(seed)
    return Qwen3_5VisionModel(tiny_vision_config()).eval()


def processor():
    from transformers import Qwen2VLImageProcessorFast
    return Qwen2VLImageProcessorFast(patch_size=16, temporal_patch_size=2, merge_size=2,
                                     size={"shortest_edge": 32 * 32, "longest_edge": 256 * 256})


# ---- 预处理 ---------------------------------------------------------------------


def test_each_image_gets_its_own_grid_and_pixel_rows():
    from PIL import Image

    data = [png((255, 0, 0), (64, 64)), png((0, 0, 255), (128, 64))]
    got = preprocess_images(processor(), data)
    assert [i.grid for i in got] == [(1, 4, 4), (1, 4, 8)]
    assert [i.tokens for i in got] == [4, 8]
    for image, raw in zip(got, data):
        alone = processor()(images=[Image.open(io.BytesIO(raw)).convert("RGB")], return_tensors="pt")
        torch.testing.assert_close(image.pixels, alone["pixel_values"], rtol=0, atol=0)


def test_images_with_transparency_or_palettes_are_read_as_rgb():
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGBA", (64, 64), (255, 0, 0, 0)).save(buf, format="PNG")
    (image,) = preprocess_images(processor(), [buf.getvalue()])
    assert image.grid == (1, 4, 4) and image.pixels.shape[-1] == 3 * 2 * 16 * 16


# ---- 前向时编码 ----------------------------------------------------------------


def pixel_batch(m, tok, d, seed=0):
    """两行: 第 0 行两张图 (2x2 与 1x2 个占位符), 第 1 行一张 (1x2)."""
    from dataclasses import replace

    from test_decision import examples
    g = torch.Generator().manual_seed(seed)
    images = [fake_image(grid) for grid in [(1, 4, 4), (1, 2, 4), (1, 2, 4)]]
    images = [replace(i, pixels=torch.randn(i.pixels.shape[0], 1536, generator=g)) for i in images]
    a, b = examples()
    exs = [replace(a, query=f"red {IMAGE_MARK} {IMAGE_MARK} context", context_label="", images=tuple(images[:2])),
           replace(b, query=f"long {IMAGE_MARK} blue", context_label="", images=(images[2],))]
    return candidate_batch(tok, d, exs), images


def vision_model():
    from test_qwen35 import tiny_qwen35

    from sors.core.decision import DecisionConfig
    from sors.core.model import prepare_model

    tok, d, train_ids = vision_tokenizer()
    cfg = DecisionConfig(kind="candidate", blocks=0, dim=16, heads=4, option_batch_size=2)
    m = prepare_model(tiny_qwen35(), train_ids, 4, 8, 0., trainable="decision-only", decision=cfg).eval()
    attach_vision(m, tiny_visual(), tok.convert_tokens_to_ids(VISION_TOKENS[1]))
    return m, tok, d


@pytest.mark.parametrize("mode", ["off", "on"])
def test_pixels_are_encoded_like_running_the_encoder_on_each_image_alone(mode):
    m, tok, d = vision_model()
    data, images = pixel_batch(m, tok, d)
    with torch.no_grad():
        alone = [m.visual(i.pixels, grid_thw=torch.tensor([i.grid])).pooler_output for i in images]
    rows = [torch.cat(alone[:2]), alone[2]]
    embeds = torch.zeros(2, 6, 64)
    for r, e in enumerate(rows):
        embeds[r, :len(e)] = e
    with torch.no_grad():
        want = reference_logits(m, {**data, "image_embeds": embeds})
        m.set_candidate_prefix_cache(mode)
        got = m.forward_batch(data)
    valid = data["slot_ids"] >= 0
    torch.testing.assert_close(got.gather(1, data["slot_ids"].clamp_min(0))[valid],
                               want.gather(1, data["slot_ids"].clamp_min(0))[valid], atol=1e-5, rtol=1e-4)


def test_the_encoder_stays_frozen_while_the_backbone_trains():
    from sors.core.model import trainable_param_groups

    m, tok, d = vision_model()
    for p in m.base.parameters():
        p.requires_grad_(True)
    data, _ = pixel_batch(m, tok, d)
    m.train().set_candidate_prefix_cache("off")
    loss = m.forward_batch(data).gather(1, data["slot_ids"].clamp_min(0))[data["slot_ids"] >= 0].sum()
    loss.backward()
    assert all(not p.requires_grad and p.grad is None for p in m.visual.parameters())
    grouped = {id(p) for g in trainable_param_groups(m, 1e-4, 1e-3) for p in g["params"]}
    assert not grouped & {id(p) for p in m.visual.parameters()}
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.base.parameters())


def test_pixels_without_an_encoder_are_rejected():
    m, tok, d = vision_model()
    data, _ = pixel_batch(m, tok, d)
    del m.visual
    with pytest.raises(ValueError, match="encoder"):
        m.forward_batch(data)


# ---- 从底座权重载入 -----------------------------------------------------------


def test_the_encoder_loads_from_a_multimodal_base_checkpoint():
    from test_qwen35 import tiny_qwen35
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    from sors.core.vision import load_vision

    text = tiny_qwen35().config.to_dict()
    torch.manual_seed(4)
    full = Qwen3_5ForConditionalGeneration(Qwen3_5Config(text_config=text, vision_config=tiny_vision_config().to_dict(),
                                                         image_token_id=60))
    with tempfile.TemporaryDirectory() as directory:
        full.save_pretrained(directory)
        visual, image_token_id = load_vision(Path(directory), dtype=torch.float32, device="cpu", local_files_only=True)
    assert image_token_id == 60
    want = full.model.visual.state_dict()
    got = visual.state_dict()
    assert set(got) == set(want)
    for k in want:
        torch.testing.assert_close(got[k], want[k], rtol=0, atol=0)
    assert not any(p.requires_grad for p in visual.parameters())
