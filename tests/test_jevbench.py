"""sors.data.jevbench 的测试: JevBench 的三档公开题 (下载到 data/jevbench) 读成部署形态的评估集.

跑:  PYTHONPATH=src .venv/bin/python tests/test_jevbench.py
"""

import json
import pathlib
import shutil
import tempfile
import urllib.request

from pydantic import TypeAdapter

from _runner import run
from sors.core.prompt import state_text
from sors.data.jevbench import COMMIT, DEFAULT_DIR, TIERS, load_jevbench, to_eval_example
from sors.serve.api import Question
from sors.serve.menus import to_example


def _rows(tier: str, data_dir=DEFAULT_DIR) -> list[dict]:
    with (pathlib.Path(data_dir) / f"{tier}.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_three_public_tiers_with_the_published_counts():
    sets = load_jevbench()
    assert TIERS == ("easy", "original", "hard")
    assert list(sets) == ["jevbench_easy", "jevbench_original", "jevbench_hard"]
    assert [len(sets[f"jevbench_{t}"]) for t in TIERS] == [48, 72, 111]


def test_each_question_is_the_menu_the_server_builds_for_that_request():
    """state 与 question 原样当成一次 /v1/systemone 请求, 走推理服务同一条路 (serve.menus.to_example, 不带上下文标签);
    评估集与之只差 gold."""
    sets = load_jevbench()
    for t in TIERS:
        for r, ex in zip(_rows(t), sets[f"jevbench_{t}"]):
            want = to_example(TypeAdapter(Question).validate_python(r["question"]), r["state"], "")
            assert (ex.query, ex.option_names, ex.options, ex.question, ex.qtype, ex.context_label, ex.codes) == \
                   (want.query, want.option_names, want.options, want.question, want.qtype, "", None), r["id"]


def test_choice_rows_follow_the_criteria_order_the_benchmark_sends():
    """JevBench 把 criteria 对象原样发出去, 菜单就按它的键序排 (文件里是字母序), 而非 labels 的顺序."""
    ex = load_jevbench()["jevbench_easy"][0]
    assert ex.option_names == [
        "billing_question: Asks about a charge, invoice or payment",
        "cancel_order: Wants to cancel an order",
        "change_address: Wants to change the delivery address",
        "report_damage: Received an item that is broken or damaged",
        "track_order: Wants to know where an order is or when it arrives",
    ]
    assert ex.gold_idx == 4 and ex.query == "Where is my package? I ordered it last week and it still hasn't arrived."


def test_gold_row_is_the_expected_label():
    sets = load_jevbench()
    for t in TIERS:
        for r, ex in zip(_rows(t), sets[f"jevbench_{t}"]):
            q, gold_name = r["question"], ex.option_names[ex.gold_idx]
            assert ex.label == ex.options[ex.gold_idx], r["id"]
            if q["type"] == "choice":
                assert gold_name.split(":")[0] == r["expected"], (r["id"], gold_name)
            elif q["type"] == "noul":
                assert ex.qtype == "bool" and gold_name.split(":")[0] == r["expected"], (r["id"], gold_name)
            else:
                assert ex.gold_idx == r["expected"] and gold_name == q["criteria"][r["expected"]], r["id"]


def test_each_question_carries_its_family():
    """family 原样带上, 评估时按它分开报. 三档各自的 family 集合与文件相同."""
    sets = load_jevbench()
    for t in TIERS:
        for r, ex in zip(_rows(t), sets[f"jevbench_{t}"]):
            assert ex.family == r["family"], r["id"]
    assert {ex.family for ex in sets["jevbench_hard"]} == {
        "long_policy", "multi_hop", "judge_hard", "temporal_numeric", "probability", "trap", "ambiguous", "tradeoff",
        "adversarial", "routing_hard"}


def test_hard_questions_mark_the_surface_answer_as_the_decoy():
    """hard 档的 provenance.surface_answer 是出题人埋的诱饵 (只看表面会选的那个错答案), 记成菜单上那一行的类 id.
    choice 按 criteria 的键, noul 按 no / yes, score 按分级序号 (文件里写成字符串). 没写诱饵的题 decoy 为 None;
    easy / original 没有诱饵."""
    sets = load_jevbench()
    hard = list(zip(_rows("hard"), sets["jevbench_hard"]))
    for r, ex in hard:
        surface = r["provenance"].get("surface_answer")
        if surface is None or (r["question"]["type"] == "choice" and surface not in r["question"]["criteria"]):
            assert ex.decoy is None, r["id"]
            continue
        row = ex.options.index(ex.decoy)
        assert row != ex.gold_idx, r["id"]
        name = ex.option_names[row]
        assert (row == int(surface)) if r["question"]["type"] == "score" else name.split(":")[0] == surface, r["id"]
    assert sum(ex.decoy is not None for _, ex in hard) == 106
    assert all(ex.decoy is None for t in ("easy", "original") for ex in sets[f"jevbench_{t}"])


def test_json_states_are_written_out_as_indented_json():
    hard = [r for r in _rows("hard") if isinstance(r["state"], dict)]
    assert hard, "the hard tier has object states"
    by_query = {ex.query for ex in load_jevbench()["jevbench_hard"]}
    for r in hard:
        assert state_text(r["state"]) in by_query and state_text(r["state"]).startswith("{\n  "), r["id"]


def test_a_label_set_that_disagrees_with_the_menu_is_refused():
    """labels 与菜单对不上 (多一项、少一项) 就报错, 不拿错位的 gold 评估."""
    r = _rows("easy")[0]
    r["labels"] = r["labels"] + ["refund_request"]
    try:
        to_eval_example(r)
    except ValueError as e:
        assert "easy-intent-00" in str(e), e
        return
    raise AssertionError("a labels list with an option the criteria lack was accepted")


def test_the_files_live_under_data_which_git_ignores():
    """题目文件不进仓库: 放在 data/jevbench, data/ 整个在 .gitignore 里."""
    root = pathlib.Path(__file__).resolve().parents[1]
    assert DEFAULT_DIR == root / "data" / "jevbench"
    assert "/data/" in (root / ".gitignore").read_text().splitlines()


def test_a_missing_tier_is_downloaded_from_the_pinned_commit():
    """缺哪档就从 JevBench 钉死的 commit 下哪档, 已有的不再下."""
    urls = []

    def fake(url, dest):
        urls.append(url)
        shutil.copy(DEFAULT_DIR / pathlib.Path(url).name, dest)

    with tempfile.TemporaryDirectory() as d:
        for t in ("easy", "original"):
            shutil.copy(DEFAULT_DIR / f"{t}.jsonl", d)
        real, urllib.request.urlretrieve = urllib.request.urlretrieve, fake
        try:
            sets = load_jevbench(d)
        finally:
            urllib.request.urlretrieve = real
        assert (pathlib.Path(d) / "hard.jsonl").exists()
    assert urls == [f"https://raw.githubusercontent.com/fstandhartinger/jevbench/{COMMIT}/datasets/public/hard.jsonl"]
    assert len(sets["jevbench_hard"]) == 111


def test_a_file_that_differs_from_the_pinned_sha256_is_refused():
    with tempfile.TemporaryDirectory() as d:
        for t in TIERS:
            shutil.copy(DEFAULT_DIR / f"{t}.jsonl", d)
        rows = _rows("easy", d)
        rows[0]["expected"] = "cancel_order"
        (pathlib.Path(d) / "easy.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        try:
            load_jevbench(d)
        except RuntimeError as e:
            assert "sha256" in str(e) and "easy.jsonl" in str(e), e
            return
    raise AssertionError("an edited easy.jsonl was accepted")


if __name__ == "__main__":
    run(globals())
