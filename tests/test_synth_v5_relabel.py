"""Relabelled bindings (the data's relabel/<domain>.json, v5.20): dropped bindings are not drawn, relabelled ones
carry a single answer, and with fallback_correct the 'insufficient information' option is the answer whenever the
draw adds it.

Run:  PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/test_synth_v5_relabel.py
"""

import json
import random
import shutil

from sors.data.paths import asset_path
from sors.data.synth_v5 import item_example, load_synth_v5
from test_synth_v5 import HOTEL_C, HOTEL_Q, TELECOM_Q, _data

MISSING = {"id": "telecom_idv#0", "label": "Call log", "goal": "ambiguous",
           "text": "PIN matched. The date-of-birth result was lost when the screen froze.",
           "questions": [{"question": "direct", "answer": {"no": 0.5, "yes": 0.5},
                          "uncertainty": "insufficient_evidence", "why": "The second check is lost.",
                          "fallback": "unknown"},
                         {"question": "detail_cancel_contract", "answer": {"no": 0.5, "yes": 0.5},
                          "uncertainty": "judgment", "why": "Reads two ways."}]}
RELABEL = {"domain": "telecom", "bindings": [
    {"context": "telecom_idv#0", "question": "direct", "answer": "no", "fallback": "unknown"},
    {"context": "telecom_idv#0", "question": "detail_cancel_contract", "drop": True}]}


def _items(relabel=True):
    bank = [q for q in TELECOM_Q if q["id"] in ("direct", "detail_cancel_contract")]
    d = _data(telecom=(bank, [MISSING]), hotel=(HOTEL_Q, HOTEL_C))
    shutil.copy(asset_path("synth-intents-v5.25/schema.py"), d)
    if relabel:
        (d / "relabel").mkdir()
        (d / "relabel" / "telecom.json").write_text(json.dumps(RELABEL))
    try:
        return load_synth_v5(d)
    finally:
        shutil.rmtree(d)


def test_a_dropped_binding_is_not_drawn():
    assert {it.question_id for it in _items() if it.domain == "telecom"} == {"direct"}


def test_a_relabelled_binding_answers_with_the_single_label_without_the_fallback():
    it = next(it for it in _items() if it.question_id == "direct")
    assert it.target is None
    ex = item_example(it, random.Random(0), 0.0, fallback_rate=0.0)
    assert ex.option_names[ex.gold_idx].startswith("no")


def test_the_insufficient_information_option_is_the_answer_whenever_it_is_shown():
    it = next(it for it in _items() if it.question_id == "direct")
    ex = item_example(it, random.Random(0), 0.0, fallback_rate=1.0)
    assert ex.option_names[ex.gold_idx].startswith("insufficient_information")
    assert ex.target is None


def test_without_the_overlay_the_distribution_stays():
    it = next(it for it in _items(relabel=False) if it.question_id == "direct")
    assert it.target == [0.5, 0.5]
