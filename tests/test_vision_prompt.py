"""state 里的图片: data URI 解析、提示里的占位符展开、candidate 批里的网格与像素 (CPU).

图在 query 里是一个 IMAGE_MARK 字符, MenuExample.images 依次给出这些图预处理后的样子.
提示里每个标记展开成 <|vision_start|> + N 个 <|image_pad|> + <|vision_end|>, N = 时间 x 高/2 x 宽/2.
"""

import base64
import io
from dataclasses import replace

import pytest
import torch
from test_candidate import candidate_batch
from test_decision import examples

from sors.core.menu import MenuExample
from sors.core.vision import IMAGE_MARK, VISION_TOKENS, VisionImage, split_data_uris


def png(color=(255, 0, 0), size=(32, 32)) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def uri(data: bytes, kind="png") -> str:
    return f"data:image/{kind};base64,{base64.b64encode(data).decode()}"


def fake_image(grid=(1, 4, 4), fill=0.0, channels=6):
    t, h, w = grid
    return VisionImage(grid=grid, pixels=torch.full((t * h * w, channels), fill))


def vision_tokenizer():
    """测试用 tokenizer, 先装三个视觉 token 再装 D / 类型 / 上下文 token (真实 Qwen3.5 自带视觉 token, 也排在前面):
    小模型的词表只扩到放开的那些行为止, 视觉 token 要落在里面."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    from sors.core.tokens import (
        install_context_tokens,
        install_d_tokens,
        install_type_tokens,
    )

    words = ["[PAD]", "[UNK]", "red", "blue", "green", "short", "long", "context", "Question",
             "Answer", "Options", "Option", "Which", "color", "Customer", "message", ":", ".", "?"]
    backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]")
    tok.add_tokens(VISION_TOKENS, special_tokens=True)
    d = install_d_tokens(tok)
    return tok, d, d + install_type_tokens(tok) + install_context_tokens(tok)


# ---- data URI -------------------------------------------------------------------


def test_data_uris_become_marks_in_place_and_their_bytes_come_out_in_order():
    a, b = png((255, 0, 0)), png((0, 0, 255))
    text, images = split_data_uris(f"before {uri(a)} middle {uri(b, 'jpeg')} after")
    assert text == f"before {IMAGE_MARK} middle {IMAGE_MARK} after"
    assert images == [a, b]


def test_data_uris_inside_json_state_keep_the_surrounding_json():
    a = png()
    text, images = split_data_uris(f'{{\n  "photo": "{uri(a)}",\n  "note": "hi"\n}}')
    assert text == f'{{\n  "photo": "{IMAGE_MARK}",\n  "note": "hi"\n}}'
    assert images == [a]


def test_text_without_data_uris_is_unchanged():
    assert split_data_uris("data:text/plain;base64,aGk= stays") == ("data:text/plain;base64,aGk= stays", [])


@pytest.mark.parametrize("text", [f"a {IMAGE_MARK} b", f"{uri(b'not an image')}"])
def test_forged_marks_and_undecodable_images_are_rejected(text):
    with pytest.raises(ValueError):
        split_data_uris(text)


# ---- 提示 -----------------------------------------------------------------------


def test_menu_example_needs_one_image_per_mark():
    with pytest.raises(ValueError, match="image"):
        MenuExample(f"a {IMAGE_MARK} b", [1, 2], 0, 1, ["x", "y"], images=())
    with pytest.raises(ValueError, match="image"):
        MenuExample("a b", [1, 2], 0, 1, ["x", "y"], images=(fake_image(),))


def test_each_mark_expands_to_its_placeholder_run_between_vision_tokens():
    from sors.core.prompt import encode_prompts, prompt_pieces

    tok, _, _ = vision_tokenizer()
    ex = MenuExample(f"red {IMAGE_MARK} blue {IMAGE_MARK}", [1, 2], 0, 1, ["red", "blue"],
                     context_label="", images=(fake_image((1, 4, 4)), fake_image((1, 2, 4))))
    ctx, _ = prompt_pieces(ex)
    got = encode_prompts(tok, [ctx])[0]
    vs, pad, ve = tok.convert_tokens_to_ids(VISION_TOKENS)
    red, blue = tok.convert_tokens_to_ids(["red", "blue"])
    assert got == [red, vs, *[pad] * 4, ve, blue, vs, *[pad] * 2, ve]


def test_vision_token_names_written_in_the_state_stay_plain_text():
    from sors.core.prompt import encode_prompts, prompt_pieces

    tok, _, _ = vision_tokenizer()
    ex = MenuExample("red <|image_pad|>", [1, 2], 0, 1, ["red", "blue"], context_label="")
    got = encode_prompts(tok, [prompt_pieces(ex)[0]])[0]
    assert tok.convert_tokens_to_ids("<|image_pad|>") not in got


# ---- candidate 批 ---------------------------------------------------------------


def test_candidate_batch_carries_each_rows_grids_and_pixels():
    tok, d, _ = vision_tokenizer()
    a, b = examples()
    exs = [replace(a, query=f"red {IMAGE_MARK} {IMAGE_MARK}", images=(fake_image((1, 4, 4), 1.0), fake_image((1, 2, 4), 2.0))),
           replace(b, query="long blue context")]
    data = candidate_batch(tok, d, exs)
    assert data["image_grids"].tolist() == [[[1, 4, 4], [1, 2, 4]], [[0, 0, 0], [0, 0, 0]]]
    pixels = data["image_pixels"]
    assert pixels.shape == (2, 16 + 8, 6)
    assert pixels[0, :16].eq(1).all() and pixels[0, 16:].eq(2).all() and pixels[1].eq(0).all()
    pad = tok.convert_tokens_to_ids("<|image_pad|>")
    for col in range(3):
        ids = data["option_input_ids"][0, col][data["option_attention_mask"][0, col].bool()]
        assert int((ids == pad).sum()) == 4 + 2


def test_batches_without_images_have_no_image_fields():
    tok, d, _ = vision_tokenizer()
    assert not {"image_grids", "image_pixels"} & set(candidate_batch(tok, d, examples()))


@pytest.mark.parametrize("architecture", ["slots", "minimal", "structural"])
def test_other_architectures_refuse_examples_with_images(architecture):
    from sors.core.batch import collate

    tok, d, _ = vision_tokenizer()
    ex = replace(examples()[0], query=f"red {IMAGE_MARK}", images=(fake_image(),))
    with pytest.raises(ValueError, match="image"):
        collate([ex], tok, d, 4, architecture=architecture)
