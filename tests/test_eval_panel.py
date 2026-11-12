"""sors.evaluation.panel 的测试: 评估面板的题目格式转换与计分.

面板把各来源的评估集统一成「一条 /v1/systemone 请求 + 每道题的标准答案」, 经 HTTP 打分,
于是任何 checkpoint、任何版本的代码、原版模型都用同一套题和同一套计分.

跑:  .venv/bin/pytest tests/test_eval_panel.py
"""

import pytest

from sors.core.menu import LabeledSet
from sors.evaluation.panel import (item_from_simple, item_from_suite_row, item_from_task, items_from_labeled,
                                   run_items, score_question, summarize)


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
    assert {k: s[k] for k in ("top1", "metric", "chance")} == {"top1": True, "metric": 1.0, "chance": 0.25}
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


def _items(n):
    return [item_from_task("sors", {**_task(), "id": f"t-{i}"}) for i in range(n)]


def _lines(path):
    import json
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_run_writes_a_start_line_before_and_a_done_line_after_each_request(tmp_path):
    out = tmp_path / "results.jsonl"
    seen = []

    def post(request):
        seen.append(request)
        return {"answers": {"q": {"choice": "option_b", "probabilities": {}}}}

    run_items(_items(2), post, out, workers=1)
    lines = _lines(out)
    assert [(x["event"], x["id"]) for x in lines] == [("start", "t-0"), ("done", "t-0"), ("start", "t-1"), ("done", "t-1")]
    done = lines[1]
    assert done["set"] == "sors" and done["scores"]["q"]["top1"] is True and done["scores"]["q"]["type"] == "choice"
    assert "ms" in done and seen[0] == _items(1)[0]["request"]


def test_a_failed_request_is_recorded_with_its_types_and_the_run_goes_on(tmp_path):
    out = tmp_path / "results.jsonl"

    def post(request):
        raise RuntimeError("HTTP 422: too long")

    run_items(_items(2), post, out, workers=1)
    done = [x for x in _lines(out) if x["event"] == "done"]
    assert len(done) == 2 and "HTTP 422" in done[0]["error"] and done[0]["type"] == {"q": "choice"}
    assert summarize(done)["sors"]["errors"] == 2


def test_rerun_skips_answered_items_and_retries_failed_ones(tmp_path):
    out = tmp_path / "results.jsonl"
    calls = []
    run_items(_items(2), lambda r: (_ for _ in ()).throw(RuntimeError("down")), out, workers=1)

    def post(request):
        calls.append(request)
        return {"answers": {"q": {"choice": "option_a", "probabilities": {}}}}

    run_items(_items(3), post, out, workers=1)
    assert len(calls) == 3  # 两条报错的重试, 一条新的
    run_items(_items(3), post, out, workers=1)
    assert len(calls) == 3  # 都答过了, 不再发


def test_report_reads_the_last_done_line_of_each_item(tmp_path):
    from sors.evaluation.panel import load_results
    out = tmp_path / "results.jsonl"
    run_items(_items(1), lambda r: (_ for _ in ()).throw(RuntimeError("down")), out, workers=1)
    run_items(_items(1), lambda r: {"answers": {"q": {"choice": "option_b", "probabilities": {}}}}, out, workers=1)
    rows = load_results(out)
    assert len(rows) == 1 and "error" not in rows[0]


def test_each_question_records_confidence_nll_and_brier_of_its_distribution():
    s = score_question("choice", 3, {"choice": "B", "probabilities": {"A": .2, "B": .5, "C": .3}}, "A")
    assert s["conf"] == pytest.approx(.5) and s["nll"] == pytest.approx(-__import__("math").log(.2))
    assert s["brier"] == pytest.approx(.8 ** 2 + .5 ** 2 + .3 ** 2)
    n = score_question("noul", 2, {"noul": 0.9}, "no")
    assert n["conf"] == pytest.approx(.9) and n["p_yes"] == pytest.approx(.9)
    assert n["brier"] == pytest.approx(.9 ** 2 + .9 ** 2)
    sc = score_question("score", 3, {"score": 1.5, "probabilities": {"0": .1, "1": .3, "2": .6}}, 2)
    assert sc["conf"] == pytest.approx(.6) and sc["nll"] == pytest.approx(-__import__("math").log(.6))


def test_summarize_reports_calibration_and_the_noul_decision_rate():
    def q(t, top1, conf, nll, brier, p_yes=None):
        d = {"type": t, "top1": top1, "metric": float(top1), "chance": .5, "conf": conf, "nll": nll, "brier": brier}
        if p_yes is not None:
            d["p_yes"] = p_yes
        return d
    rows = [{"set": "s", "family": None, "scores": {"q": q("noul", True, .95, .05, .005, .95)}},
            {"set": "s", "family": None, "scores": {"q": q("noul", False, .9, 2.3, 1.62, .1)}},
            {"set": "s", "family": None, "scores": {"q": q("noul", True, .6, .5, .32, .6)}},
            {"set": "s", "family": None, "scores": {"q": q("choice", True, .7, .36, .14)}}]
    out = summarize(rows)["s"]
    assert out["noul_decisive"] == pytest.approx(2 / 3)          # 0.95 和 0.1 落在 0.2..0.8 之外
    assert out["noul_decisive_acc"] == pytest.approx(1 / 2)      # 决断的两道里对一道
    assert out["conf"] == pytest.approx((.95 + .9 + .6 + .7) / 4)
    assert out["nll"] == pytest.approx((.05 + 2.3 + .5 + .36) / 4)
    assert out["brier"] == pytest.approx((.005 + 1.62 + .32 + .14) / 4)
    # ECE: 10 个等宽置信度分箱 (sors.evaluation.metrics.ece), |箱内正确率 - 箱内平均置信度| 按题数加权;
    # 0.95 与 0.9 同在 0.9..1.0 箱: 正确率 0.5, 平均置信度 0.925
    assert out["ece"] == pytest.approx((2 * abs(.5 - .925) + abs(1 - .6) + abs(1 - .7)) / 4)
