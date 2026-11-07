"""请求里的图片: TypeSafe System One 的 assets (https://docs.typesafe.ai/images).

图片列在请求顶层的 assets 里, 键是调用方取的引用串; 引用串写在哪, 图就在哪. 这个服务只在 state 里换图:
模型只在 state 那段 (所有题共用的前缀) 读图, 选项分支只有文字. instructions / criteria 里出现引用串拒收
(ImagesOutsideState), 而非把图当文字留着. 也只收 base64 JSON: 带 part 的 multipart 资产拒收.

state 里怎么换 (规范原话的做法): 只看字符串值, 对象的键原样是文字; 整个值恰好是一个引用串的, 整个值就是那张图;
别的字符串里从左往右换, 不重叠, 同一处能配上几个引用时取最长的. 每出现一次就多一份图, 依次排在返回的列表里,
顺序与 state 渲染成文字 (core.prompt.state_text) 后标记出现的顺序相同 (JSON 照插入顺序深度优先写出).
每个资产都必须被引用.

图片的限制照 Jev 的: 只收 PNG / JPEG 静态图, format 与文件一致, 宽高各 280..1568 像素 (两端都算),
单张解码后不超过 16 MiB; 引用串各不超过 1024 字节 (UTF-8), 合计不超过 65536; 展开后的请求
(state 与问题的文字, 加上每一次引用各一份图) 不超过 32 MiB. 超出上限是 ImageTooLarge (HTTP 413),
别的毛病是 ImageRejected (HTTP 400), 与官方的状态码相同. 超限的图拒收, 不缩放.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import re

from sors.core.prompt import state_text
from sors.core.vision import IMAGE_MARK
from sors.serve.api import Asset
from sors.serve.engine import UnsupportedInput

MIN_SIDE, MAX_SIDE = 280, 1568
MAX_IMAGE_BYTES = 16 * 2**20
MAX_REFERENCE_BYTES = 1024
MAX_REFERENCES_BYTES = 65536
MAX_EXPANDED_BYTES = 32 * 2**20

_BASE64 = re.compile(r"[A-Za-z0-9+/]*={0,2}")
_PIL_FORMATS = {"PNG": "png", "JPEG": "jpg"}


class ImageRejected(ValueError):
    """资产不合规: 格式、编码、文件本身、尺寸下限、没被引用、空引用. HTTP 400."""


class ImageTooLarge(ValueError):
    """超出大小上限: 单张文件、宽高、引用串长度、展开后的请求. HTTP 413."""


class ImagesOutsideState(ValueError):
    """引用串出现在 instructions / criteria 里; 这个服务的模型只在 state 里读图."""


def _strings(value):
    """value 里所有的字符串值 (对象的键不算)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def _question_texts(q):
    yield from _strings(q.instructions)
    if q.type == "noul":
        if q.criteria is not None:
            yield from _strings([q.criteria.true, q.criteria.false])
    else:
        yield from _strings(q.criteria if q.type == "score" else list(q.criteria.values()))


def _check_references(refs) -> None:
    total = 0
    for ref in refs:
        if not ref.strip():
            raise ImageRejected("asset references must not be empty or only whitespace")
        size = len(ref.encode())
        if size > MAX_REFERENCE_BYTES:
            raise ImageTooLarge(f"the asset reference {ref[:32]!r}... is {size} bytes; "
                                f"each reference takes at most {MAX_REFERENCE_BYTES}")
        total += size
    if total > MAX_REFERENCES_BYTES:
        raise ImageTooLarge(f"the asset references come to {total} bytes; together they take at most "
                            f"{MAX_REFERENCES_BYTES}")


def _decode(ref: str, a: Asset) -> bytes:
    from PIL import Image, UnidentifiedImageError

    if a.part is not None:
        raise ImageRejected(f"asset {ref!r} names a multipart part; this server takes images only as "
                            "data_base64 in a JSON body, not multipart")
    if a.data_base64 is None:
        raise ImageRejected(f"asset {ref!r} has no data_base64")
    if a.format is None:
        raise ImageRejected(f"asset {ref!r} has data_base64 but no format")
    s = a.data_base64
    if len(s) % 4 or not _BASE64.fullmatch(s):
        raise ImageRejected(f"asset {ref!r}: data_base64 must be standard base64 with = padding, "
                            "no whitespace and no data: prefix")
    try:
        data = base64.b64decode(s, validate=True)
    except binascii.Error as exc:
        raise ImageRejected(f"asset {ref!r}: data_base64 is not valid base64: {exc}") from exc
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageTooLarge(f"asset {ref!r} is {len(data)} bytes; each image file takes at most {MAX_IMAGE_BYTES}")
    try:
        with Image.open(io.BytesIO(data)) as img:
            kind = _PIL_FORMATS.get(img.format)
            animated = getattr(img, "is_animated", False)
            width, height = img.size
            img.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageRejected(f"asset {ref!r} is not a readable PNG or JPEG file: {exc}") from exc
    if kind is None:
        raise ImageRejected(f"asset {ref!r} is {img.format}; only PNG and JPEG are taken")
    if kind != ("jpg" if a.format == "jpeg" else a.format):
        raise ImageRejected(f"asset {ref!r} says format {a.format!r} but the file is {kind}")
    if animated:
        raise ImageRejected(f"asset {ref!r} is an animated image; send one still frame")
    if min(width, height) < MIN_SIDE:
        raise ImageRejected(f"asset {ref!r} is {width}x{height}; each side needs at least {MIN_SIDE} pixels")
    if max(width, height) > MAX_SIDE:
        raise ImageTooLarge(f"asset {ref!r} is {width}x{height}; each side takes at most {MAX_SIDE} pixels")
    return data


def resolve(state, questions: dict, assets: dict[str, Asset] | None):
    """(state, questions, assets) -> (换好图片标记的 state, 标记依次对应的图片字节).

    state 的形状不变 (字符串还是字符串, 对象还是对象), 只是引用串换成了 vision.IMAGE_MARK.
    state 里本来就有 IMAGE_MARK 时是 UnsupportedInput: 那是服务内部的图片位置, 调用方伪造不得.
    """
    if any(IMAGE_MARK in s for s in _strings(state)):
        raise UnsupportedInput("the state contains the reserved image mark U+E000")
    if not assets:
        return state, []
    _check_references(assets)
    for qid, q in questions.items():
        for text in _question_texts(q):
            for ref in assets:
                if ref in text:
                    raise ImagesOutsideState(f"question {qid!r} refers to asset {ref!r}; this server reads "
                                             "images only in the state, not in instructions or criteria")
    data = {ref: _decode(ref, a) for ref, a in assets.items()}
    pattern = re.compile("|".join(re.escape(r) for r in sorted(assets, key=len, reverse=True)))
    images: list[bytes] = []
    used: set[str] = set()

    def use(ref: str) -> str:
        images.append(data[ref])
        used.add(ref)
        return IMAGE_MARK

    def walk(v):
        if isinstance(v, str):
            return use(v) if v in data else pattern.sub(lambda m: use(m.group(0)), v)
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x) for x in v]
        return v

    out = walk(state)
    for ref in data:
        if ref not in used:
            raise ImageRejected(f"asset {ref!r} is not referenced in the state")
    text = len(state_text(out).encode()) + len(json.dumps([q.model_dump() for q in questions.values()]).encode())
    expanded = text + sum(len(i) for i in images)
    if expanded > MAX_EXPANDED_BYTES:
        raise ImageTooLarge(f"the expanded request (text plus one copy of an image per reference) is {expanded} "
                            f"bytes; this server takes at most {MAX_EXPANDED_BYTES}")
    return out, images
