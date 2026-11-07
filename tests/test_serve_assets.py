"""请求里的图片 (TypeSafe System One 的 assets): 引用串换成图片标记, 图片照 Jev 的限制校验.

规范见 https://docs.typesafe.ai/images: 图片列在顶层 assets 里, 键是引用串, 写在哪图就在哪.
这个服务只收 state 里的引用 (模型只在 state 那段读图), 只收 base64 JSON (不收 multipart).
"""

import base64
import io

import pytest

from sors.core.vision import IMAGE_MARK
from sors.serve import assets as A
from sors.serve.api import SystemOneRequest
from sors.serve.assets import ImageRejected, ImagesOutsideState, ImageTooLarge, resolve
from sors.serve.engine import UnsupportedInput

NOUL = {"ok": {"type": "noul", "instructions": "Ok?"}}


def png(color=(255, 0, 0), size=(280, 280)) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def jpg(size=(280, 280)) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, (0, 0, 255)).save(buf, format="JPEG")
    return buf.getvalue()


def asset(data: bytes, fmt="png") -> dict:
    return {"type": "image", "format": fmt, "data_base64": base64.b64encode(data).decode()}


def run(state, assets, questions=NOUL):
    req = SystemOneRequest(state=state, model="m", questions=questions, assets=assets)
    return resolve(req.state, req.questions, req.assets)


# ---- 引用换成图片 -----------------------------------------------------------------


def test_a_whole_json_value_becomes_the_image_and_keys_stay_text():
    red = png()
    state, images = run({"door-1": "door-1", "note": "hi"}, {"door-1": asset(red)})
    assert state == {"door-1": IMAGE_MARK, "note": "hi"}
    assert images == [red]


def test_references_inside_sentences_are_replaced_and_each_use_adds_the_image_again():
    red, blue = png(), jpg()
    state, images = run("first R then B then R", {"R": asset(red), "B": asset(blue, "jpg")})
    assert state == f"first {IMAGE_MARK} then {IMAGE_MARK} then {IMAGE_MARK}"
    assert images == [red, blue, red]


def test_images_come_out_in_the_order_the_state_is_written():
    red, blue = png(), jpg()
    state, images = run({"frames": [{"f": "B"}, {"f": "R"}], "last": "R"}, {"R": asset(red), "B": asset(blue, "jpeg")})
    assert state == {"frames": [{"f": IMAGE_MARK}, {"f": IMAGE_MARK}], "last": IMAGE_MARK}
    assert images == [blue, red, red]


def test_a_whole_value_match_wins_over_a_shorter_reference_inside_it():
    red, blue = png(), jpg()
    state, images = run(["img-1-x", "see img-1-x", "img-1"], {"img-1": asset(red), "img-1-x": asset(blue, "jpg")})
    assert state == [IMAGE_MARK, f"see {IMAGE_MARK}", IMAGE_MARK]
    assert images == [blue, blue, red]


def test_a_request_without_assets_is_unchanged():
    assert run({"a": "b"}, None) == ({"a": "b"}, [])
    assert run("text", {}) == ("text", [])


def test_the_reserved_mark_in_the_state_is_refused():
    with pytest.raises(UnsupportedInput):
        run(f"a {IMAGE_MARK}", None)


# ---- 引用的规则 -------------------------------------------------------------------


def test_an_asset_referenced_nowhere_is_rejected():
    with pytest.raises(ImageRejected, match="unused"):
        run("R", {"R": asset(png()), "unused": asset(png())})


@pytest.mark.parametrize("questions", [
    {"q": {"type": "noul", "instructions": "Is it R?"}},
    {"q": {"type": "choice", "instructions": "Which?", "criteria": {"a": "like R", "b": None}}},
    {"q": {"type": "noul", "instructions": "Ok?", "criteria": {"true": {"examples": ["R"]}, "false": "no"}}},
    {"q": {"type": "score", "instructions": "How much?", "criteria": ["low", "R"]}},
])
def test_references_in_instructions_or_criteria_are_refused(questions):
    with pytest.raises(ImagesOutsideState, match="R"):
        run("R", {"R": asset(png())}, questions)


@pytest.mark.parametrize("ref", ["", "   "])
def test_empty_references_are_rejected(ref):
    with pytest.raises(ImageRejected):
        run(f"x{ref}x", {ref: asset(png())})


def test_a_reference_over_1024_utf8_bytes_is_too_large():
    ref = "图" * 342  # 342 个字, 1026 字节
    with pytest.raises(ImageTooLarge):
        run(ref, {ref: asset(png())})
    ok = "图" * 341  # 1023 字节
    assert run(ok, {ok: asset(png())})[0] == IMAGE_MARK


def test_all_references_together_over_65536_bytes_are_too_large():
    refs = [f"{i:04d}" + "x" * 1020 for i in range(65)]  # 65 x 1024 = 66560 字节
    with pytest.raises(ImageTooLarge):
        run(" ".join(refs), {r: asset(png()) for r in refs})


# ---- 图片的规则 -------------------------------------------------------------------


def test_multipart_parts_are_rejected():
    with pytest.raises(ImageRejected, match="multipart"):
        run("R", {"R": {"type": "image", "part": "photo"}})


@pytest.mark.parametrize("encode", [
    lambda d: base64.b64encode(d).decode().rstrip("="),  # 缺 = 补位
    lambda d: "data:image/png;base64," + base64.b64encode(d).decode(),
    lambda d: base64.encodebytes(d).decode(),  # 带换行
    lambda d: base64.b64encode(d).decode()[:-4] + "ab-_",  # URL 安全字母表的字符
])
def test_base64_must_be_the_standard_padded_alphabet(encode):
    data = png()
    data += b"\0" * (1 if len(data) % 3 == 0 else 0)  # 让编码带 =, 缺补位那条才真的缺
    with pytest.raises(ImageRejected):
        run("R", {"R": {"format": "png", "data_base64": encode(data)}})


def test_base64_without_a_format_is_rejected():
    with pytest.raises(ImageRejected):
        run("R", {"R": {"data_base64": base64.b64encode(png()).decode()}})


def test_an_asset_without_data_is_rejected():
    with pytest.raises(ImageRejected):
        run("R", {"R": {"type": "image", "format": "png"}})


@pytest.mark.parametrize("data, fmt", [(jpg(), "png"), (png(), "jpg")])
def test_the_format_must_match_the_file(data, fmt):
    with pytest.raises(ImageRejected, match="format"):
        run("R", {"R": asset(data, fmt)})


def test_other_formats_are_rejected():
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (280, 280)).save(buf, format="GIF")
    with pytest.raises(ImageRejected):
        run("R", {"R": asset(buf.getvalue())})


def test_damaged_files_are_rejected():
    with pytest.raises(ImageRejected):
        run("R", {"R": asset(png()[:100])})


def test_animated_pngs_are_rejected():
    from PIL import Image
    buf = io.BytesIO()
    frames = [Image.new("RGB", (280, 280), c) for c in ((255, 0, 0), (0, 0, 255))]
    frames[0].save(buf, format="PNG", save_all=True, append_images=frames[1:])
    with pytest.raises(ImageRejected, match="animated"):
        run("R", {"R": asset(buf.getvalue())})


@pytest.mark.parametrize("size", [(279, 300), (300, 279)])
def test_a_side_under_280_pixels_is_rejected(size):
    with pytest.raises(ImageRejected, match="280"):
        run("R", {"R": asset(png(size=size))})


@pytest.mark.parametrize("size", [(1569, 300), (300, 1569)])
def test_a_side_over_1568_pixels_is_too_large(size):
    with pytest.raises(ImageTooLarge, match="1568"):
        run("R", {"R": asset(png(size=size))})


def test_both_ends_of_the_size_range_are_accepted():
    _, images = run("A B", {"A": asset(png(size=(280, 280))), "B": asset(png(size=(1568, 1568)))})
    assert len(images) == 2


def test_a_file_over_the_per_image_limit_is_too_large(monkeypatch):
    data = png()
    monkeypatch.setattr(A, "MAX_IMAGE_BYTES", len(data) - 1)
    with pytest.raises(ImageTooLarge):
        run("R", {"R": asset(data)})
    monkeypatch.setattr(A, "MAX_IMAGE_BYTES", len(data))
    assert run("R", {"R": asset(data)})[1] == [data]


def test_every_use_of_an_image_counts_toward_the_expanded_request(monkeypatch):
    data = png()
    monkeypatch.setattr(A, "MAX_EXPANDED_BYTES", 3 * len(data))
    run("R R", {"R": asset(data)})
    with pytest.raises(ImageTooLarge, match="expanded"):
        run("R R R R", {"R": asset(data)})
