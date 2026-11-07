"""Official vision decision sources in their released parquet schemas (offline, CPU).

Each adapter turns rows into a state with one IMAGE_MARK per image, a question and the official
menu; the image bytes ride along and become VisionImage only when Decisions.sample has a processor.
"""

import io
import json
import random

import pytest

from sors.core.vision import IMAGE_MARK, VisionImage
from sors.data import public_vision as pv


def jpeg(size=(64, 48), color=(200, 30, 30)) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def image(size=(64, 48)):
    return {"bytes": jpeg(size), "path": None}


def processor():
    from transformers import Qwen2VLImageProcessorFast
    return Qwen2VLImageProcessorFast(patch_size=16, temporal_patch_size=2, merge_size=2,
                                     size={"shortest_edge": 32 * 32, "longest_edge": 256 * 256})


def shown(ex) -> str:
    """Everything the model reads besides the pixels."""
    return "\n".join([ex.query, ex.question or "", *ex.option_names])


# ---- A-OKVQA -------------------------------------------------------------------


def _aokvqa(**kw):
    row = {"image": image(), "question_id": "q1", "question": "What is the man awaiting?",
           "choices": ["skateboarder", "train", "delivery", "cab"], "correct_choice_idx": 3,
           "direct_answers": "['cab', 'taxi']", "difficult_direct_answer": False,
           "rationales": ["He has bags and waits on the road for a cab."]}
    return {**row, **kw}


def test_aokvqa_keeps_the_official_menu_and_hides_rationales():
    out = pv.aokvqa([_aokvqa()])
    (ex,) = out.examples
    assert ex.option_names == ["skateboarder", "train", "delivery", "cab"] and ex.gold_idx == 3
    assert ex.query.count(IMAGE_MARK) == 1 and len(ex.images) == 1 and isinstance(ex.images[0], bytes)
    assert ex.question == "What is the man awaiting?"
    text = ex.query + ex.question
    assert "taxi" not in text and "bags" not in text
    assert out.ids == ["q1"]


def test_aokvqa_skips_menus_with_repeated_options():
    out = pv.aokvqa([_aokvqa(choices=["cab", "train", "cab", "bus"])])
    assert not out.examples and out.skipped["duplicate_options"] == 1


# ---- ScienceQA ------------------------------------------------------------------


def _science(**kw):
    row = {"image": image(), "question": "Which of these states is farthest north?",
           "choices": ["West Virginia", "Louisiana"], "answer": 0, "hint": "Look at the map.",
           "task": "closed choice", "grade": "grade2", "subject": "social science", "topic": "geography",
           "category": "Geography", "skill": "Read a map",
           "lecture": "Maps have four cardinal directions.", "solution": "West Virginia is farthest north."}
    return {**row, **kw}


def test_scienceqa_reads_the_hint_and_leaves_lecture_and_solution_out():
    out = pv.scienceqa([_science(), _science(image=None), _science(hint="")])
    assert out.skipped["no_image"] == 1 and len(out.examples) == 2
    a, b = out.examples
    assert "Look at the map." in a.query and a.query.count(IMAGE_MARK) == 1
    assert b.query == IMAGE_MARK
    for ex in out.examples:
        assert "cardinal" not in shown(ex) and "farthest north." not in ex.query
        assert ex.option_names[ex.gold_idx] == "West Virginia"


# ---- The Cauldron letter MCQ (AI2D, IconQA) -------------------------------------


def _cauldron(user, assistant, n_images=1):
    return {"images": [image() for _ in range(n_images)],
            "texts": [{"user": u, "assistant": a, "source": "fixture"} for u, a in zip(user, assistant)]}


MCQ = "Question: What do respiration and combustion give out\nChoices:\nA. Oxygen\nB. Carbon dioxide\nC. Heat\nAnswer with the letter."


def test_cauldron_letter_choices_become_the_menu_and_the_letter_the_answer():
    out = pv.cauldron_mcq([_cauldron([MCQ, "How many sides does this shape have?"], ["Answer: B", "4"])], "ai2d")
    (ex,) = out.examples
    assert ex.question == "What do respiration and combustion give out"
    assert ex.option_names == ["Oxygen", "Carbon dioxide", "Heat"] and ex.gold_idx == 1
    assert "Choices" not in ex.query + ex.question and ex.query.count(IMAGE_MARK) == 1
    assert out.skipped["not_multiple_choice"] == 1
    assert out.ids == ["0/0"]


def test_cauldron_letter_outside_the_menu_is_an_error():
    with pytest.raises(ValueError, match="letter"):
        pv.cauldron_mcq([_cauldron([MCQ], ["Answer: D"])], "ai2d")


# ---- The Cauldron statements (VSR, NLVR2) ---------------------------------------


def test_statement_templates_collapse_to_one_question_per_source():
    rows = [_cauldron(['Is the statement "The cake is next to the person." accurate regarding the image?\nAnswer yes or no.',
                       'Evaluate: Does the caption "The horse is on the bus." match the image?\nAnswer yes or no.'],
                      ["Yes.", "No."])]
    out = pv.cauldron_statements(rows, "vsr")
    a, b = out.examples
    assert a.qtype == "bool" and a.option_names == ["no", "yes"]
    assert (a.gold_idx, b.gold_idx) == (1, 0)
    assert a.question == 'Is this statement true of the image? "The cake is next to the person."'
    assert b.question == 'Is this statement true of the image? "The horse is on the bus."'


def test_nlvr2_pairs_put_both_images_in_the_state_left_then_right():
    user = ('The first image is the image on the left, the second image is the image on the right. '
            'Assess this claim about the two images: "There are exactly two flutes.". Correct or not? Answer yes or no.')
    (ex,) = pv.cauldron_statements([_cauldron([user], ["Yes."], n_images=2)], "nlvr2").examples
    assert ex.query.count(IMAGE_MARK) == 2 and len(ex.images) == 2
    assert json.loads(ex.query.replace(IMAGE_MARK, "<img>")) == {"left image": "<img>", "right image": "<img>"}
    assert ex.question == 'Is this statement true of the two images? "There are exactly two flutes."'


def test_statement_answers_other_than_yes_or_no_are_errors():
    with pytest.raises(ValueError, match="yes or no"):
        pv.cauldron_statements([_cauldron(['Is "A cat." right?'], ["Maybe."])], "vsr")


# ---- Multimodal-Mind2Web --------------------------------------------------------


def _cand(node, tag, box, **attrs):
    attributes = {"backend_node_id": node, "bounding_box_rect": box, **attrs}
    return json.dumps({"tag": tag, "attributes": json.dumps(attributes), "backend_node_id": node,
                       "is_original_target": True, "is_top_level_target": True})


def _web(**kw):
    row = {"action_uid": "a1", "annotation_id": "t1", "confirmed_task": "rent a car in Brooklyn",
           "operation": json.dumps({"op": "CLICK", "value": ""}),
           "pos_candidates": [_cand("7", "button", "10,20,80,30", aria_label="Search cars")],
           "neg_candidates": [_cand("3", "div", "0,0,64,48"), _cand("9", "a", "5,40,30,200"),
                              _cand("5", "a", "0,60,20,10")],
           "screenshot": image((64, 48)), "action_reprs": ["[heading] CAR -> CLICK", "[button] Search -> CLICK"],
           "target_action_index": "1", "target_action_reprs": "[button] Search -> CLICK"}
    return {**row, **kw}


def test_mind2web_menu_is_the_candidates_on_the_screenshot_with_history_up_to_the_step():
    out = pv.mm_mind2web([_web()])
    (ex,) = out.examples
    # node 5 starts below the 48-pixel-tall screenshot; node 9 reaches into it
    assert [n.split(":")[0] for n in ex.option_names] == ["3", "7", "9"]
    assert ex.option_names[ex.gold_idx].startswith("7: button")
    state = json.loads(ex.query.replace(IMAGE_MARK, "<img>"))
    assert state == {"task": "rent a car in Brooklyn", "screenshot": "<img>",
                     "previous_actions": ["[heading] CAR -> CLICK"]}
    assert "is_original_target" not in shown(ex)
    assert out.ids == ["t1/a1"]


def test_mind2web_targets_below_the_screenshot_and_ambiguous_steps_are_counted():
    below = _web(pos_candidates=[_cand("7", "button", "10,60,80,30")])
    two = _web(pos_candidates=[_cand("7", "button", "1,1,5,5"), _cand("8", "a", "1,1,5,5")])
    alone = _web(neg_candidates=[_cand("5", "a", "0,60,20,10")])
    out = pv.mm_mind2web([below, two, alone])
    assert not out.examples
    assert out.skipped == {"target_off_screenshot": 1, "multiple_positives": 1, "one_option": 1}


def test_crop_keeps_the_top_of_the_page_at_its_width():
    from PIL import Image
    row = _web(screenshot=image((64, 300)), raw_html="<html/>", cleaned_html="<html/>")
    out = pv.crop_mind2web_row(row, height=100)
    assert Image.open(io.BytesIO(out["screenshot"]["bytes"])).size == (64, 100)
    assert "raw_html" not in out and "cleaned_html" not in out
    short = _web()
    assert pv.crop_mind2web_row(short, height=100)["screenshot"]["bytes"] == short["screenshot"]["bytes"]


def test_nlvr2_subsample_is_deterministic_and_keeps_rows_with_their_images():
    rows = [_cauldron([f'Is "s{i}-{j}" right?' for j in range(3)], ["Yes."] * 3) for i in range(5)]
    a = pv.sample_cauldron_questions(rows, 7, seed=0)
    b = pv.sample_cauldron_questions(rows, 7, seed=0)
    assert a == b and sum(len(r["texts"]) for r in a) == 7
    assert all(r["texts"] and r["images"] for r in a)


# ---- Decisions.sample turns bytes into VisionImage -----------------------------


def test_sampling_preprocesses_images_and_refuses_without_a_processor():
    out = pv.aokvqa([_aokvqa()])
    with pytest.raises(ValueError, match="processor"):
        out.sample(1, random.Random(0))
    out.processor = processor()
    (ex,) = out.sample(1, random.Random(0))
    assert len(ex.images) == 1 and isinstance(ex.images[0], VisionImage)
    assert ex.images[0].grid == (1, 4, 4)
    assert isinstance(out.examples[0].images[0], bytes)  # stored bytes stay bytes
    assert out.summary()["images"] == 1


# ---- manifest loading -----------------------------------------------------------


def write_parquet(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(rows), path)


def vision_manifest(root):
    from sors.data.public_decisions import PUBLIC_DATASETS
    assert set(pv.VISION_DATASETS) <= set(PUBLIC_DATASETS)
    sources = []

    def add(name, rows, **kw):
        path = root / f"{name}.parquet"
        write_parquet(path, rows)
        sources.append({"dataset": name, "split": "train", "path": path.name, "source": "fixture",
                        "license": "fixture-only", **kw})

    add("aokvqa", [_aokvqa(), _aokvqa(question_id="q2", correct_choice_idx=0)])
    add("ai2d", [_cauldron([MCQ], ["Answer: C"])])
    add("nlvr2", [_cauldron(['Is "Two flutes." true? Answer yes or no.'], ["No."], n_images=2)])
    add("mm-mind2web", [_web()])
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"version": 1, "sources": sources}))
    return manifest


def test_manifest_loads_parquet_sources_and_records_their_digest(tmp_path):
    import hashlib

    from sors.data.public_decisions import load_public_decisions
    manifest = vision_manifest(tmp_path)
    out = load_public_decisions("aokvqa", manifest)
    assert len(out.examples) == 2 and out.has_images
    assert out.ids[0].startswith("aokvqa/aokvqa.parquet/")
    digest = hashlib.sha256((tmp_path / "aokvqa.parquet").read_bytes()).hexdigest()
    assert out.summary()["sources"][0]["files"] == [{"path": str(tmp_path / "aokvqa.parquet"), "sha256": digest}]
    for name, items in [("ai2d", 1), ("nlvr2", 1), ("mm-mind2web", 1)]:
        assert load_public_decisions(name, manifest).summary()["items"] == items


# ---- train CLI --------------------------------------------------------------------


def _train_cli():
    import importlib.util
    import pathlib
    script = pathlib.Path(__file__).resolve().parents[1] / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("vision_train_cli", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_train_cli_samples_vision_sources_through_the_base_models_image_processor(tmp_path, monkeypatch):
    from sors.core import vision
    cli = _train_cli()
    manifest = vision_manifest(tmp_path)
    asked = []
    monkeypatch.setattr(vision, "load_processor",
                        lambda model, max_pixels=None, **kw: asked.append((model, max_pixels)) or processor())
    args = cli.build_parser().parse_args(["--dataset", "aokvqa+nlvr2+contractnli", "--public-manifest", str(manifest),
                                          "--eval", "simple", "--model", "Qwen/Qwen3.5-0.8B-Base",
                                          "--max-image-pixels", "4096"])
    with pytest.raises(SystemExit, match="contractnli"):  # the fixture manifest declares no text source
        cli.build_data(args)
    args.dataset = "aokvqa+nlvr2"
    asked.clear()
    sample, _, info = cli.build_data(args)
    assert asked == [("Qwen/Qwen3.5-0.8B-Base", 4096)]  # one processor shared by every vision source
    assert info["vision"] == {"model": "Qwen/Qwen3.5-0.8B-Base", "max_image_pixels": 4096}
    batch = sample(4, random.Random(0))
    assert all(ex.images and all(isinstance(i, VisionImage) for i in ex.images) for ex in batch)
    assert {len(ex.images) for ex in batch} == {1, 2}


def test_train_cli_text_sources_load_no_image_processor(tmp_path, monkeypatch):
    from sors.core import vision
    cli = _train_cli()
    monkeypatch.setattr(vision, "load_processor", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("loaded")))
    root = tmp_path
    (root / "c.json").write_text(json.dumps({"labels": {"h": {"hypothesis": "Two years."}}, "documents": [
        {"id": 1, "text": "Two years.", "annotation_sets": [{"annotations": {"h": {"choice": "Entailment"}}}]}]}))
    (root / "m.json").write_text(json.dumps({"version": 1, "sources": [
        {"dataset": "contractnli", "split": "train", "path": "c.json", "source": "f", "license": "f"}]}))
    args = cli.build_parser().parse_args(["--dataset", "contractnli", "--public-manifest", str(root / "m.json"),
                                          "--eval", "simple"])
    _, _, info = cli.build_data(args)
    assert "vision" not in info


def test_training_attaches_the_frozen_encoder_of_the_vision_model(monkeypatch):
    import types

    import torch
    from test_vision_encoder import tiny_visual
    from test_vision_prompt import vision_tokenizer

    from sors.core import vision
    cli = _train_cli()
    tok, _, _ = vision_tokenizer()
    pad = tok.convert_tokens_to_ids("<|image_pad|>")
    read = []
    monkeypatch.setattr(vision, "load_vision", lambda model, **kw: read.append(model) or (tiny_visual(), pad))
    m = torch.nn.Linear(2, 2)
    args = types.SimpleNamespace(vision_model=None, model="Qwen/Qwen3.5-0.8B-Base")
    cli.attach_training_vision(m, tok, args)
    assert read == ["Qwen/Qwen3.5-0.8B-Base"] and m.image_token_id == pad
    assert not any(p.requires_grad for p in m.visual.parameters())
    monkeypatch.setattr(vision, "load_vision", lambda model, **kw: (tiny_visual(), pad + 1))
    with pytest.raises(SystemExit, match="token"):
        cli.attach_training_vision(torch.nn.Linear(2, 2), tok, args)
