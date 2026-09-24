"""decidophobia.simple_eval 的测试: datasets/synth-simple-eval 读成按菜单长度分开的评估集.

跑:  PYTHONPATH=src .venv/bin/python tests/test_simple_eval.py
"""

import re

from _runner import run
from decidophobia.simple_eval import SIZES, load_simple_eval


def _says(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text.lower()) is not None


def test_ten_menu_questions_at_each_size_then_ten_binary():
    sets = load_simple_eval()
    assert SIZES == (5, 10, 20, 40, 60, 100, 255)
    assert list(sets) == [f"simple{k}" for k in SIZES] + ["simple_bool"]
    for k in SIZES:
        assert len(sets[f"simple{k}"]) == 10 and {len(e.options) for e in sets[f"simple{k}"]} == {k}
    assert len(sets["simple_bool"]) == 10


def test_menu_options_are_distinct_single_lowercase_nouns():
    for k in SIZES:
        for e in load_simple_eval()[f"simple{k}"]:
            assert len(set(e.option_names)) == k and len(set(e.options)) == k
            assert all(re.fullmatch(r"[a-z]+", n) for n in e.option_names), e.option_names
            assert e.qtype == "choice" and e.context_label == "Customer message" and e.codes is None


def test_each_message_names_its_answer_and_no_other_option():
    for k in SIZES:
        for e in load_simple_eval()[f"simple{k}"]:
            gold = e.option_names[e.gold_idx]
            assert e.options[e.gold_idx] == e.label
            assert _says(e.query, gold), (e.query, gold)
            others = [n for i, n in enumerate(e.option_names) if i != e.gold_idx]
            assert not any(_says(e.query, n) or _says(e.question, n) for n in others), e.query


def test_gold_rows_are_spread_evenly_down_each_menu():
    """每档 10 题的正确答案等距铺开, 第一行和最后一行都有: 255 项那档是 D0, D28, ..., D254."""
    for k in SIZES:
        got = [e.gold_idx for e in load_simple_eval()[f"simple{k}"]]
        assert got == [round(i * (k - 1) / 9) for i in range(10)], (k, got)
    assert [e.gold_idx for e in load_simple_eval()["simple255"]][:3] == [0, 28, 56]


def test_binary_asks_whether_the_customer_wants_a_thing_half_yes_and_no_first_on_half():
    """选项 0 = no, 1 = yes (与 BoolQ 同); 5 题 yes 问的就是消息里那样, 5 题 no 问另一样; no 排第一的也是 5 题."""
    exs = load_simple_eval()["simple_bool"]
    assert sum(e.label for e in exs) == 5
    assert sum(e.options[0] == 0 for e in exs) == 5
    for e in exs:
        assert e.qtype == "bool" and dict(zip(e.options, e.option_names)) == {0: "no", 1: "yes"}
        asked = re.fullmatch(r"does the customer want ([a-z]+)\?", e.question).group(1)
        assert _says(e.query, asked) == (e.label == 1), (e.query, e.question, e.label)


if __name__ == "__main__":
    run(globals())
