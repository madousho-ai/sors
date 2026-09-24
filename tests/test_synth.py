"""decidophobia.synth 的测试: datasets/synth-intents 读成菜单题与二元题两个 LabeledSet.

跑:  PYTHONPATH=src .venv/bin/python tests/test_synth.py
"""

import collections

from _runner import run
from decidophobia.synth import load_synth, load_synth_binary


def test_synth_loads_4096_intents_with_three_utterances_each():
    s, _ = load_synth()
    assert len(s.names) == 4096
    assert len(s.queries) == 12288 and len(s.labels) == 12288
    assert collections.Counter(s.labels) == {c: 3 for c in range(4096)}
    assert s.context_label == "Customer message" and s.questions is None and s.qtype == "choice"


def test_synth_class_ids_follow_sorted_intent_ids_and_names_are_descriptions():
    """类 id 按 intent id 字母序 (与 banking77 / massive 同一约定); 菜单里显示的是描述, 同一领域内不重复."""
    s, domains = load_synth()
    assert domains[0] == "automotive" and domains[4095] == "utilities"
    assert all(n == n.lower() and 3 <= len(n.split()) <= 12 for n in s.names.values())
    per_domain = collections.defaultdict(set)
    for c, n in s.names.items():
        per_domain[domains[c]].add(n)
    assert all(len(v) == 256 for v in per_domain.values())


def test_synth_domains_are_16_of_256_intents():
    _, domains = load_synth()
    assert len(domains) == 4096 and len(set(domains)) == 16
    assert collections.Counter(domains) == {d: 256 for d in set(domains)}


def test_synth_binary_asks_each_message_its_intents_question_with_no_yes_options():
    """二元题与菜单题同一批消息、同一顺序; 选项 0 = no, 1 = yes, 与 BoolQ 同形, 上下文仍是 Customer message."""
    s, _ = load_synth()
    b = load_synth_binary()
    assert b.queries == s.queries
    assert b.names == {0: "no", 1: "yes"} and b.qtype == "bool" and b.context_label == "Customer message"
    assert len(b.questions) == 12288 and all(q.endswith("?") and q == q.lower() for q in b.questions)
    assert 0.45 <= sum(b.labels) / len(b.labels) <= 0.55


def test_synth_binary_labels_are_the_intents_answers_in_message_order():
    """telecom_report_dropped_calls 的三条消息答案是 [false, true, false], 问句三条相同."""
    s, _ = load_synth()
    b = load_synth_binary()
    c = next(c for c, n in s.names.items() if n == "report that calls keep dropping")
    rows = [i for i, lab in enumerate(s.labels) if lab == c]
    assert [b.labels[i] for i in rows] == [0, 1, 0]
    assert {b.questions[i] for i in rows} == {"does the customer mention their signal strength?"}


if __name__ == "__main__":
    run(globals())
