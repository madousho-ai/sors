"""推理服务读请求里的图片 (CPU, 随机的小 candidate 模型 + 一层视觉编码器).

请求格式照 TypeSafe System One 的 images: 图片列在顶层 assets 里, state 里写它的引用串 (serve.assets).
引擎收到的 state 里引用串已换成图片标记, 另给标记依次对应的图片字节.
"""

import base64
import tempfile

import pytest
import torch
from fastapi.testclient import TestClient
from test_vision_encoder import processor, vision_model
from test_vision_prompt import png, uri, vision_tokenizer

from sors.core.vision import IMAGE_MARK
from sors.serve import assets as A
from sors.serve.api import SystemOneRequest
from sors.serve.app import create_app
from sors.serve.engine import Engine, UnsupportedInput

QUESTIONS = {"color": {"type": "choice", "instructions": "Which color?", "criteria": {"red": None, "blue": None}},
             "ok": {"type": "noul", "instructions": "Ok?"}}
RED, BLUE = (255, 0, 0), (0, 0, 255)


def questions():
    return SystemOneRequest(state="x", model="m", questions=QUESTIONS).questions


def engine(m=None, **kw):
    if m is None:
        m, tok, d = vision_model()
    else:
        tok, d, _ = vision_tokenizer()
    return Engine(m, tok, d, processor=kw.pop("processor", processor()), **kw)


def asset(color, size=(280, 280)) -> dict:
    return {"type": "image", "format": "png", "data_base64": base64.b64encode(png(color, size)).decode()}


def body(state, assets=None, qs=QUESTIONS) -> dict:
    return {"state": state, "model": "m", "questions": qs, **({"assets": assets} if assets else {})}


# ---- 引擎 -------------------------------------------------------------------------


def test_the_state_image_changes_the_answer():
    e = engine()
    red = e.evaluate(f"red {IMAGE_MARK} context", questions(), [png(RED, (64, 64))]).probs
    blue = e.evaluate(f"red {IMAGE_MARK} context", questions(), [png(BLUE, (64, 64))]).probs
    assert not torch.allclose(torch.tensor(red["color"]), torch.tensor(blue["color"]), atol=1e-4)


def test_an_image_counts_its_placeholders_as_input_tokens():
    e = engine()
    plain = e.evaluate("red context", questions()).input_tokens
    with_image = e.evaluate(f"red {IMAGE_MARK} context", questions(), [png(RED, (64, 64))]).input_tokens
    # 64x64 是 2x2 个占位符, 加 vision_start / vision_end; 决策模型每道题各读一次 state, 两道题各算一次
    assert with_image - plain == len(QUESTIONS) * (4 + 2)


def test_images_in_json_state_values_are_read_in_place():
    e = engine()
    image = [png(RED, (64, 64))]
    as_json = e.evaluate({"photo": IMAGE_MARK, "note": "red"}, questions(), image).probs
    same_text = e.evaluate(f'{{\n  "photo": "{IMAGE_MARK}",\n  "note": "red"\n}}', questions(), image).probs
    assert as_json == same_text


def test_data_uris_in_the_state_are_plain_text():
    state, _ = engine().prompts(f"red {uri(png(RED, (64, 64)))}", questions())["color"]
    assert "<|image_pad|>" not in state and "base64," in state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the packed path's DeltaNet kernels need CUDA")
def test_packed_models_answer_image_requests_like_the_shared_prefix_cache():
    m, _, _ = vision_model()
    e = engine(m.cuda())
    image = [png(RED, (64, 64))]
    want = e.evaluate(f"red {IMAGE_MARK}", questions(), image).probs
    e.lm.set_candidate_prefix_cache("packed")
    got = e.evaluate(f"red {IMAGE_MARK}", questions(), image).probs
    for qid in want:
        torch.testing.assert_close(torch.tensor(got[qid]), torch.tensor(want[qid]), atol=1e-5, rtol=1e-4)


def test_a_model_without_a_vision_encoder_refuses_images_but_still_answers_text():
    m, _, _ = vision_model()
    del m.visual
    e = engine(m)
    with pytest.raises(UnsupportedInput, match="image"):
        e.evaluate(f"red {IMAGE_MARK}", questions(), [png(RED, (64, 64))])
    assert set(e.evaluate("red", questions()).probs) == set(QUESTIONS)


def test_prompt_previews_show_images_as_placeholder_runs():
    prompts = engine().prompts(f"red {IMAGE_MARK}", questions(), [png(RED, (64, 64))])
    state, _ = prompts["color"]
    assert state.count("<|image_pad|>") == 4 and "<|vision_start|>" in state


# ---- HTTP -------------------------------------------------------------------------


def test_the_http_api_reads_assets_referenced_in_the_state():
    client = TestClient(create_app(engine(), "m"))
    red = client.post("/v1/systemone", json=body({"photo": "ref-1"}, {"ref-1": asset(RED)}))
    blue = client.post("/v1/systemone", json=body({"photo": "ref-1"}, {"ref-1": asset(BLUE)}))
    assert red.status_code == 200 and blue.status_code == 200
    assert red.json()["answers"]["color"] != blue.json()["answers"]["color"]


@pytest.mark.parametrize("assets, status", [
    ({"ref-1": asset(RED), "spare": asset(RED)}, 400),  # 有一个没被引用
    ({"ref-1": asset(RED, (200, 280))}, 400),  # 边长不到 280
    ({"ref-1": asset(RED, (1600, 280))}, 413),  # 边长超过 1568
])
def test_rejected_assets_get_the_official_status_codes(assets, status):
    client = TestClient(create_app(engine(), "m"))
    r = client.post("/v1/systemone", json=body("see ref-1", assets))
    assert r.status_code == status


def test_references_in_questions_are_422_on_questions():
    client = TestClient(create_app(engine(), "m"))
    qs = {"q": {"type": "noul", "instructions": "Is it like ref-1?"}}
    r = client.post("/v1/systemone", json=body("ref-1", {"ref-1": asset(RED)}, qs))
    assert r.status_code == 422 and r.json()["detail"][0]["loc"] == ["body", "questions"]


def test_the_reserved_mark_and_images_for_a_text_model_are_422_on_state():
    client = TestClient(create_app(engine(), "m"))
    forged = client.post("/v1/systemone", json=body(f"a {IMAGE_MARK}"))
    assert forged.status_code == 422 and forged.json()["detail"][0]["loc"] == ["body", "state"]
    m, _, _ = vision_model()
    del m.visual
    text_only = TestClient(create_app(engine(m), "m")).post("/v1/systemone", json=body("ref-1", {"ref-1": asset(RED)}))
    assert text_only.status_code == 422 and text_only.json()["detail"][0]["loc"] == ["body", "state"]


def test_a_body_over_the_limit_is_413(monkeypatch):
    client = TestClient(create_app(engine(), "m"))
    small = body("red")
    monkeypatch.setattr(A, "MAX_BODY_BYTES", 10)
    assert client.post("/v1/systemone", json=small).status_code == 413


def test_prompt_previews_read_assets_too(tmp_path):
    client = TestClient(create_app(engine(), "m", demo_dir=tmp_path))
    r = client.post("/demo/prompts", json=body("see ref-1", {"ref-1": asset(RED)}))
    assert r.status_code == 200 and "<|image_pad|>" in r.json()["prompts"]["color"]["state"]


# ---- 自带视觉编码器的完整模型目录 ---------------------------------------------------


def test_an_exported_model_carries_its_vision_encoder_and_reads_images_without_a_base(tmp_path):
    from unittest.mock import patch

    from sors.core.pretrained import save_pretrained
    from sors.serve.engine import load_engine

    e = engine()
    image = [png(RED, (64, 64))]
    want = e.evaluate(f"red {IMAGE_MARK}", questions(), image).probs
    save_pretrained(e.lm, e.tok, {"layout": "context-first"}, tmp_path / "release", processor=e.processor)
    with patch("sors.core.vision.load_vision", side_effect=AssertionError("a base model was read")):
        loaded = load_engine(tmp_path / "release", device="cpu", dtype=torch.float32, local_files_only=True)
    assert loaded.evaluate(f"red {IMAGE_MARK}", questions(), image).probs == want
    with pytest.raises(ValueError, match="vision"):
        load_engine(tmp_path / "release", device="cpu", vision_model="Qwen/Qwen3.5-2B-Base")


def test_the_bundled_processor_still_takes_the_pixel_cap(tmp_path):
    from sors.core.pretrained import save_pretrained
    from sors.core.vision import preprocess_images
    from sors.serve.engine import load_engine

    e = engine()
    save_pretrained(e.lm, e.tok, {"layout": "context-first"}, tmp_path / "release", processor=e.processor)
    capped = load_engine(tmp_path / "release", device="cpu", max_image_pixels=64 * 64)
    (big,) = preprocess_images(capped.processor, [png(RED, (256, 256))])
    t, h, w = big.grid
    assert t * h * w * 16 * 16 <= 64 * 64


def test_exporting_a_vision_model_needs_its_processor(tmp_path):
    from sors.core.pretrained import save_pretrained

    e = engine()
    with pytest.raises(ValueError, match="processor"):
        save_pretrained(e.lm, e.tok, {"layout": "context-first"}, tmp_path / "release")
    assert not (tmp_path / "release").exists()


def test_the_processor_loads_with_a_pixel_cap():
    from sors.core.vision import load_processor, preprocess_images

    with tempfile.TemporaryDirectory() as directory:
        processor().save_pretrained(directory)
        capped = load_processor(directory, max_pixels=64 * 64, local_files_only=True)
    (big,) = preprocess_images(capped, [png((255, 0, 0), (256, 256))])
    t, h, w = big.grid
    assert t * h * w * 16 * 16 <= 64 * 64
