"""sors.evaluation.panel 的测试: 评估面板的题目格式转换与计分.

面板把各来源的评估集统一成「一条 /v1/systemone 请求 + 每道题的标准答案」, 经 HTTP 打分,
于是任何 checkpoint、任何版本的代码、原版模型都用同一套题和同一套计分.

跑:  .venv/bin/pytest tests/test_eval_panel.py
"""

import pytest

from sors.core.menu import LabeledSet
from sors.evaluation.panel import (item_from_simple, item_from_suite_row, item_from_task, items_from_labeled,
                                   score_question, summarize)


def _task(qtype="choice", expected="option_b"):
    q = {"type": "choice", "instructions": "Which?", "criteria": {"option_a": "A", "option_b": "B", "option_c": "C"}}
    labels = ["option_a", "option_b", "option_c"]
    if qtype == "noul":
        q = {"type": "noul", "instructions": "Is it?", "criteria": {"true": "yes it is", "false": "no it is not"}}
        labels = ["no", "yes"]
    if qtype == "score":
        q = {"type": "score", "instructions": "How bad?", "criteria": ["fine", "bad", "awful"]}
        labels = ["0", "1", "2"]
    return {"id": "t-1", "family": "fam", "state": {"msg": "hi"}, "question": q, "labels": labels,
            "expected": expected, "split": "private", "group": "t-1"}


def test_a_jevbench_format_task_becomes_one_request_with_one_question():
    item = item_from_task("sors", _task())
    assert item["set"] == "sors" and item["id"] == "t-1" and item["family"] == "fam"
    assert item["request"] == {"state": {"msg": "hi"}, "questions": {"q": _task()["question"]}}
    assert item["gold"] == {"q": "option_b"} and item["type"] == {"q": "choice"} and item["k"] == {"q": 3}


def test_the_task_keeps_no_answer_fields_in_the_request():
    item = item_from_task("sors", _task())
    flat = repr(item["request"])
    assert "expected" not in flat and "provenance" not in flat and "labels" not in flat


def test_a_simple_eval_row_names_its_options_and_its_gold_by_text():
    row = {"id": "k3-00", "qtype": "choice", "context": "can I get peach?", "question": "Which item?",
           "options": ["peach", "tractor", "daisy"], "answer": 0}
    item = item_from_simple(row)
    q = item["request"]["questions"]["q"]
    assert item["request"]["state"] == "can I get peach?"
    assert q == {"type": "choice", "instructions": "Which item?", "criteria": {"peach": None, "tractor": None, "daisy": None}}
    assert item["gold"] == {"q": "peach"} and item["family"] == "k3"


def test_a_simple_bool_row_becomes_a_noul_question():
    row = {"id": "bool-03", "qtype": "bool", "context": "The shop is closed on Sunday.", "question": "Is it open on Sunday?",
           "options": ["no", "yes"], "answer": 0}
    item = item_from_simple(row)
    assert item["type"] == {"q": "noul"} and item["gold"] == {"q": "no"}
    assert item["request"]["questions"]["q"] == {"type": "noul", "instructions": "Is it open on Sunday?"}


def test_a_labeled_set_shows_every_class_in_name_order():
    ls = LabeledSet(queries=["wake me at 7", "play jazz"], labels=[1, 0], names={0: "play_music", 1: "set_alarm"})
    items = items_from_labeled("massive", ls, instructions="What does the user want?")
    assert [it["gold"] for it in items] == [{"q": "set_alarm"}, {"q": "play_music"}]
    assert items[0]["request"] == {"state": "wake me at 7", "questions": {"q": {
        "type": "choice", "instructions": "What does the user want?", "criteria": {"play_music": None, "set_alarm": None}}}}
    assert items[0]["id"] == "massive-00000"


def test_a_labeled_bool_set_uses_its_own_question_per_row():
    ls = LabeledSet(queries=["Paris is in France."], labels=[1], names={0: "no", 1: "yes"}, context_label="Passage",
                    questions=["is paris in france?"], qtype="bool")
    item = items_from_labeled("boolq", ls)[0]
    assert item["request"]["questions"]["q"] == {"type": "noul", "instructions": "is paris in france?"}
    assert item["gold"] == {"q": "yes"}


def test_a_decision_index_row_keeps_every_question_and_its_gold():
    row = {"state": {}, "questions": {"q1": {"type": "choice", "instructions": "x", "criteria": {"A": "a", "B": "b"}},
                                      "q2": {"type": "noul", "instructions": "y"}},
           "gold": {"q1": "B", "q2": "yes"}, "_evaluation": {"run_id": "24:MMLU:1", "dataset": "MMLU"}}
    item = item_from_suite_row("dindex", row)
    assert item["id"] == "24:MMLU:1" and item["family"] == "MMLU"
    assert item["request"] == {"state": {}, "questions": row["questions"]}
    assert item["gold"] == {"q1": "B", "q2": "yes"} and item["type"] == {"q1": "choice", "q2": "noul"}
    assert item["k"] == {"q1": 2, "q2": 2}


def test_choice_is_right_when_the_chosen_key_is_gold():
    s = score_question("choice", 4, {"choice": "B", "probabilities": {"A": .1, "B": .7, "C": .1, "D": .1}}, "B")
    assert s == {"top1": True, "metric": 1.0, "chance": 0.25}
    assert score_question("choice", 4, {"choice": "A", "probabilities": {}}, "B")["metric"] == 0.0


@pytest.mark.parametrize("p, gold, metric, top1", [(0.9, "yes", 1.0, True), (0.6, "yes", 0.0, True),
                                                   (0.1, "no", 1.0, True), (0.3, "no", 0.0, True), (0.95, "no", 0.0, False)])
def test_noul_counts_as_wrong_inside_the_undecided_band(p, gold, metric, top1):
    s = score_question("noul", 2, {"noul": p}, gold)
    assert s["metric"] == metric and s["top1"] is top1


def test_score_is_one_minus_the_normalized_distance_of_the_expected_level():
    ans = {"score": 1.5, "probabilities": {"0": .1, "1": .3, "2": .6}}
    s = score_question("score", 3, ans, 2)
    assert s["metric"] == pytest.approx(1 - 0.5 / 2) and s["top1"] is True
    assert score_question("score", 3, {"score": 0.0, "probabilities": {"0": 1, "1": 0, "2": 0}}, 2)["metric"] == 0.0


def test_summarize_reports_each_type_and_chance_corrects_choice():
    rows = [
        {"set": "s", "family": "f", "scores": {"q": {"type": "choice", "top1": True, "metric": 1.0, "chance": .25}}},
        {"set": "s", "family": "f", "scores": {"q": {"type": "choice", "top1": False, "metric": 0.0, "chance": .25}}},
        {"set": "s", "family": "f", "scores": {"q": {"type": "noul", "top1": True, "metric": 1.0, "chance": .5}}},
        {"set": "s", "family": "f", "scores": {"q": {"type": "score", "top1": False, "metric": 0.75, "chance": None}}},
        # 请求报错时它的每道题都记为答错, 按题型进各自的分母
        {"set": "s", "family": "f", "type": {"q": "choice"}, "k": {"q": 4}, "error": "HTTP 422"},
    ]
    out = summarize(rows)["s"]
    assert out["n"] == 5 and out["errors"] == 1
    assert out["choice"] == pytest.approx((1 / 3 - 0.25) / 0.75)
    assert out["noul"] == 1.0 and out["score"] == 0.75
    assert out["top1"] == pytest.approx(2 / 5)
