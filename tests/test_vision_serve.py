"""推理服务读 state 里的 data URI 图片 (CPU, 随机的小 candidate 模型 + 一层视觉编码器).

请求格式照 TypeSafe System One, 不加字段: 图片以 data:image/...;base64,... 写在 state 里 (字符串或 JSON 的值),
服务端把它解析出来, 放回它在 state 里的位置.
"""

import base64
import tempfile

import pytest
import torch
from fastapi.testclient import TestClient
from test_vision_encoder import processor, vision_model
from test_vision_prompt import png, uri, vision_tokenizer

from sors.serve.api import SystemOneRequest
from sors.serve.app import create_app
from sors.serve.engine import Engine, UnsupportedInput

QUESTIONS = {"color": {"type": "choice", "instructions": "Which color?", "criteria": {"red": None, "blue": None}},
             "ok": {"type": "noul", "instructions": "Ok?"}}


def questions():
    return SystemOneRequest(state="x", model="m", questions=QUESTIONS).questions


def engine(m=None, **kw):
    if m is None:
        m, tok, d = vision_model()
    else:
        tok, d, _ = vision_tokenizer()
    return Engine(m, tok, d, processor=kw.pop("processor", processor()), **kw)


def image(color, size=(64, 64)):
    return uri(png(color, size))


def test_the_state_image_changes_the_answer():
    e = engine()
    red = e.evaluate(f"red {image((255, 0, 0))} context", questions()).probs
    blue = e.evaluate(f"red {image((0, 0, 255))} context", questions()).probs
    assert not torch.allclose(torch.tensor(red["color"]), torch.tensor(blue["color"]), atol=1e-4)


def test_an_image_counts_its_placeholders_as_input_tokens():
    e = engine()
    plain = e.evaluate("red context", questions()).input_tokens
    with_image = e.evaluate(f"red {image((255, 0, 0))} context", questions()).input_tokens
    # 64x64 是 2x2 个占位符, 加 vision_start / vision_end; 决策模型每道题各读一次 state, 两道题各算一次
    assert with_image - plain == len(QUESTIONS) * (4 + 2)


def test_images_in_json_state_values_are_read_in_place():
    e = engine()
    as_json = e.evaluate({"photo": image((255, 0, 0)), "note": "red"}, questions()).probs
    same_text = e.evaluate(f'{{\n  "photo": "{image((255, 0, 0))}",\n  "note": "red"\n}}', questions()).probs
    assert as_json == same_text


def test_packed_models_read_images_through_the_shared_prefix_at_inference():
    e = engine()
    want = e.evaluate(f"red {image((255, 0, 0))}", questions()).probs
    e.lm.set_candidate_prefix_cache("packed")
    got = e.evaluate(f"red {image((255, 0, 0))}", questions()).probs
    for qid in want:
        torch.testing.assert_close(torch.tensor(got[qid]), torch.tensor(want[qid]), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("state", ["a \ue000 b", "data:image/png;base64," + base64.b64encode(b"nope").decode()])
def test_forged_marks_and_broken_images_are_unsupported_input(state):
    with pytest.raises(UnsupportedInput):
        engine().evaluate(state, questions())


def test_a_model_without_a_vision_encoder_refuses_images_but_still_answers_text():
    m, _, _ = vision_model()
    del m.visual
    e = engine(m)
    with pytest.raises(UnsupportedInput, match="image"):
        e.evaluate(f"red {image((255, 0, 0))}", questions())
    assert set(e.evaluate("red", questions()).probs) == set(QUESTIONS)


def test_the_http_api_answers_images_and_rejects_unsupported_ones_with_422():
    client = TestClient(create_app(engine(), "m"))
    ok = client.post("/v1/systemone", json={"state": f"red {image((255, 0, 0))}", "model": "m", "questions": QUESTIONS})
    assert ok.status_code == 200 and set(ok.json()["answers"]) == set(QUESTIONS)
    bad = client.post("/v1/systemone", json={"state": "a \ue000", "model": "m", "questions": QUESTIONS})
    assert bad.status_code == 422 and bad.json()["detail"][0]["loc"] == ["body", "state"]


def test_prompt_previews_show_images_as_placeholder_runs():
    prompts = engine().prompts(f"red {image((255, 0, 0))}", questions())
    state, _ = prompts["color"]
    assert "base64" not in state
    assert state.count("<|image_pad|>") == 4 and "<|vision_start|>" in state


def test_the_processor_loads_with_a_pixel_cap():
    from sors.core.vision import load_processor, preprocess_images

    with tempfile.TemporaryDirectory() as directory:
        processor().save_pretrained(directory)
        capped = load_processor(directory, max_pixels=64 * 64, local_files_only=True)
    (big,) = preprocess_images(capped, [png((255, 0, 0), (256, 256))])
    t, h, w = big.grid
    assert t * h * w * 16 * 16 <= 64 * 64
