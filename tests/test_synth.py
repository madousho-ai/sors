"""decidophobia.synth 的测试: datasets/synth-intents 读成菜单题与二元题两个 LabeledSet.

跑:  PYTHONPATH=src .venv/bin/python tests/test_synth.py
"""

import collections

from _runner import run
from decidophobia.synth import load_synth


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


if __name__ == "__main__":
    run(globals())
