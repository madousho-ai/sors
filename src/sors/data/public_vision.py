"""Official vision decision sources: images in the state, official menus, hard labels.

Rows come as pyarrow's to_pylist() of the released parquet files; an image is {"bytes", "path"}.
The state holds one core.vision.IMAGE_MARK per image and the example carries the encoded bytes;
Decisions.sample preprocesses them with its processor. Explanations, rationales and annotation
flags stay out of the state. crop_mind2web_row and sample_cauldron_questions prepare compact
local files (scripts/prepare-vision-decisions.py); the adapters themselves never download.
"""

from __future__ import annotations

import io
import json
import random
import re

from sors.core.vision import IMAGE_MARK
from sors.data.public_decisions import VISION_DATASETS, Decisions, _integer, _text  # noqa: F401 (VISION_DATASETS re-exported)

MIND2WEB_VIEWPORT = 720  # first screen of a 1280-wide page
# 99.9% of candidates render under 500 characters; the rest reach 100k (huge name attributes)
MIND2WEB_OPTION_CHARS = 2000


def _bytes(image, name: str) -> bytes | None:
    if image is None or image.get("bytes") is None:
        return None
    if not isinstance(image["bytes"], bytes) or not image["bytes"]:
        raise ValueError(f"{name}: expected encoded image bytes")
    return image["bytes"]


def _distinct(names, out: Decisions) -> bool:
    """A usable official menu: no blank option, no repeated option; others are counted and skipped."""
    if any(not isinstance(n, str) or not n.strip() for n in names):
        out.skipped["blank_option"] += 1
        return False
    if len(set(names)) != len(names):
        out.skipped["duplicate_options"] += 1
        return False
    return True


def aokvqa(rows: list[dict]) -> Decisions:
    """A-OKVQA multiple choice: the photo is the whole state; rationales and direct answers stay out."""
    out = Decisions()
    for row in rows:
        data = _bytes(row["image"], "A-OKVQA image")
        if data is None:
            raise ValueError(f"A-OKVQA {row['question_id']}: missing image")
        names = list(row["choices"])
        if not _distinct(names, out):
            continue
        out.add(_text(row["question_id"], "question ID"), IMAGE_MARK, row["question"], names,
                row["correct_choice_idx"], label="Photo", images=(data,))
    return out


def scienceqa(rows: list[dict]) -> Decisions:
    """ScienceQA image questions; the hint is part of the state, lecture and solution explain the answer."""
    out = Decisions()
    for i, row in enumerate(rows):
        data = _bytes(row["image"], "ScienceQA image")
        if data is None:
            out.skipped["no_image"] += 1
            continue
        names = list(row["choices"])
        if not _distinct(names, out):
            continue
        hint = (row.get("hint") or "").strip()
        state = {"image": IMAGE_MARK, "context": hint} if hint else IMAGE_MARK
        out.add(i, state, row["question"], names, row["answer"], label="Science problem", images=(data,))
    return out


_MCQ = re.compile(r"\AQuestion: (?P<q>.+?)\nChoices:\n(?P<c>(?:[A-Z]\. .*\n)+)Answer with the letter\.\Z", re.S)
_LABELS = {"ai2d": "Science diagram", "iconqa": "Picture puzzle"}


def cauldron_mcq(rows: list[dict], source: str) -> Decisions:
    """The Cauldron's letter-choice turns (AI2D, IconQA): choices become the menu, the letter the answer.
    Turns without a choice list (open answers in IconQA) are counted as not_multiple_choice."""
    out = Decisions()
    for i, row in enumerate(rows):
        images = tuple(_bytes(im, f"{source} image") for im in row["images"])
        if not images or None in images:
            raise ValueError(f"{source} row {i}: missing image")
        for j, turn in enumerate(row["texts"]):
            m = _MCQ.match(turn["user"])
            if not m:
                out.skipped["not_multiple_choice"] += 1
                continue
            opts = re.findall(r"^([A-Z])\. (.*)$", m["c"], re.M)
            if [o[0] for o in opts] != [chr(65 + k) for k in range(len(opts))]:
                raise ValueError(f"{source} {i}/{j}: choice letters out of order")
            answer = re.fullmatch(r"Answer: ([A-Z])", turn["assistant"].strip())
            if not answer or ord(answer[1]) - 65 >= len(opts):
                raise ValueError(f"{source} {i}/{j}: answer letter {turn['assistant']!r} outside the choices")
            names = [o[1].strip() for o in opts]
            if not _distinct(names, out):
                continue
            state = IMAGE_MARK if len(images) == 1 else {f"image {k + 1}": IMAGE_MARK for k in range(len(images))}
            out.add(f"{i}/{j}", state, m["q"].strip(), names, ord(answer[1]) - 65,
                    label=_LABELS.get(source, "Image"), images=images)
    return out


_STATEMENT = {"vsr": "Is this statement true of the image?", "nlvr2": "Is this statement true of the two images?"}


def cauldron_statements(rows: list[dict], source: str) -> Decisions:
    """The Cauldron's yes/no caption checks (VSR, NLVR2). Their questions use several paraphrase
    templates around one quoted statement; the statement is kept under one fixed question."""
    out = Decisions()
    for i, row in enumerate(rows):
        images = tuple(_bytes(im, f"{source} image") for im in row["images"])
        if not images or None in images:
            raise ValueError(f"{source} row {i}: missing image")
        if source == "nlvr2":
            if len(images) != 2:
                raise ValueError(f"NLVR2 row {i}: expected a left and a right image")
            state = {"left image": IMAGE_MARK, "right image": IMAGE_MARK}
        elif len(images) == 1:
            state = IMAGE_MARK
        else:
            raise ValueError(f"{source} row {i}: expected one image")
        for j, turn in enumerate(row["texts"]):
            quoted = re.findall(r"\"(.+)\"", turn["user"], re.S)
            if len(quoted) != 1:
                raise ValueError(f"{source} {i}/{j}: expected one quoted statement")
            answer = turn["assistant"].strip().rstrip(".").lower()
            if answer not in ("no", "yes"):
                raise ValueError(f"{source} {i}/{j}: expected yes or no, got {turn['assistant']!r}")
            question = f"{_STATEMENT.get(source, _STATEMENT['vsr'])} \"{quoted[0].strip()}\""
            out.add(f"{i}/{j}", state, question, ["no", "yes"], int(answer == "yes"),
                    label="Images" if len(images) > 1 else "Image", qtype="bool", images=images)
    return out


def _box(candidate: dict):
    attributes = json.loads(candidate["attributes"])
    box = attributes.get("bounding_box_rect")
    if not box:
        return None
    x, y, w, h = (float(v) for v in box.split(","))
    return x, y, w, h


def _on_screen(box, width: int, height: int) -> bool:
    if box is None:
        return False
    x, y, w, h = box
    return w > 0 and h > 0 and x < width and x + w > 0 and y < height and y + h > 0


def mm_mind2web(rows: list[dict]) -> Decisions:
    """Multimodal-Mind2Web element choice from the screenshot (cropped to the first screen by
    crop_mind2web_row). The menu is every candidate whose box overlaps the screenshot, chosen
    without looking at which is the target; steps whose target lies below it are skipped."""
    from PIL import Image

    out = Decisions()
    for row in rows:
        data = _bytes(row["screenshot"], "Mind2Web screenshot")
        if data is None:
            out.skipped["no_screenshot"] += 1
            continue
        pos = [json.loads(c) for c in row["pos_candidates"]]
        if len(pos) != 1:
            out.skipped["multiple_positives" if pos else "no_positive"] += 1
            continue
        width, height = Image.open(io.BytesIO(data)).size
        if not _on_screen(_box(pos[0]), width, height):
            out.skipped["target_off_screenshot"] += 1
            continue
        neg = [c for c in map(json.loads, row["neg_candidates"]) if _on_screen(_box(c), width, height)]
        if not neg:
            out.skipped["one_option"] += 1
            continue
        candidates = sorted(pos + neg, key=lambda c: str(c["backend_node_id"]))
        ids = [str(c["backend_node_id"]) for c in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("Mind2Web: repeated candidate IDs")
        names = [f"{c['backend_node_id']}: {c['tag']} "
                 + json.dumps(json.loads(c["attributes"]), ensure_ascii=False, sort_keys=True) for c in candidates]
        if max(map(len, names)) > MIND2WEB_OPTION_CHARS:
            out.skipped["option_too_long"] += 1
            continue
        step = _integer(row["target_action_index"], "target_action_index")
        if not 0 <= step < len(row["action_reprs"]):
            raise ValueError("Mind2Web: target_action_index outside action_reprs")
        state = {"task": _text(row["confirmed_task"], "task"), "screenshot": IMAGE_MARK,
                 "previous_actions": list(row["action_reprs"][:step])}
        out.add(f"{row['annotation_id']}/{row['action_uid']}", state,
                "Which element should be interacted with next to carry out the task?", names,
                ids.index(str(pos[0]["backend_node_id"])), label="Browser state", images=(data,))
    return out


# ---- local preparation (scripts/prepare-vision-decisions.py) -------------------

MIND2WEB_COLUMNS = ["action_uid", "annotation_id", "confirmed_task", "operation", "pos_candidates",
                    "neg_candidates", "screenshot", "action_reprs", "target_action_index", "target_action_reprs"]


def crop_mind2web_row(row: dict, height: int = MIND2WEB_VIEWPORT) -> dict:
    """The row without its HTML, the full-page screenshot cut to its top `height` pixels (JPEG).
    A screenshot already that short keeps its original bytes."""
    from PIL import Image

    out = {k: row[k] for k in MIND2WEB_COLUMNS}
    data = _bytes(row["screenshot"], "Mind2Web screenshot")
    if data is None:
        return out
    im = Image.open(io.BytesIO(data))
    if im.size[1] > height:
        buf = io.BytesIO()
        im.convert("RGB").crop((0, 0, im.size[0], height)).save(buf, format="JPEG", quality=90)
        out["screenshot"] = {"bytes": buf.getvalue(), "path": None}
    return out


def sample_cauldron_questions(rows: list[dict], n: int, seed: int) -> list[dict]:
    """n turns drawn uniformly over all rows' turns; each row keeps its images and its drawn turns."""
    turns = [(i, j) for i, row in enumerate(rows) for j in range(len(row["texts"]))]
    if n > len(turns):
        raise ValueError(f"asked for {n} questions, only {len(turns)} available")
    keep = sorted(random.Random(seed).sample(turns, n))
    picked: dict[int, list[int]] = {}
    for i, j in keep:
        picked.setdefault(i, []).append(j)
    return [{**rows[i], "texts": [rows[i]["texts"][j] for j in js]} for i, js in picked.items()]


def convert(name: str, rows: list[dict]) -> Decisions:
    if name in ("ai2d", "iconqa"):
        return cauldron_mcq(rows, name)
    if name in ("vsr", "nlvr2"):
        return cauldron_statements(rows, name)
    return {"aokvqa": aokvqa, "scienceqa": scienceqa, "mm-mind2web": mm_mind2web}[name](rows)
