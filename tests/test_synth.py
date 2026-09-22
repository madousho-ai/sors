"""decidophobia.synth 的测试: datasets/synth-intents 读成 LabeledSet.

跑:  PYTHONPATH=src .venv/bin/python tests/test_synth.py
"""

from _runner import run
from decidophobia.synth import load_synth


def test_synth_loads_512_intents_with_two_utterances_each():
    s, _ = load_synth()
    assert len(s.names) == 512
    assert len(s.queries) == 1024 and len(s.labels) == 1024
    assert all(s.labels.count(c) == 2 for c in range(512))
    assert s.context_label == "Customer message" and s.questions is None and s.qtype == "choice"


def test_synth_class_ids_follow_sorted_intent_ids_and_names_are_descriptions():
    """类 id 按 intent id 字母序 (与 banking77 / massive 同一约定); 菜单里显示的是描述, 全局不重复."""
    s, domains = load_synth()
    assert domains[0] == "automotive" and domains[511] == "utilities"
    assert all(n == n.lower() and 3 <= len(n.split()) <= 12 for n in s.names.values())
    assert len(set(s.names.values())) == 512


def test_synth_domains_are_per_class_for_holding_out():
    _, domains = load_synth()
    assert len(domains) == 512 and len(set(domains)) == 16
    assert domains.count("hotel") == 32


if __name__ == "__main__":
    run(globals())
