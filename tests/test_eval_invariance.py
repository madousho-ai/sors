"""scripts/eval-invariance.py 里不碰模型的部分: 从原菜单造变体.

跑:  PYTHONPATH=src .venv/bin/python tests/test_eval_invariance.py
"""

import importlib.util
import pathlib

from _runner import run
from decidophobia.synth import synth_eval_examples

_SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "eval-invariance.py"
_spec = importlib.util.spec_from_file_location("eval_invariance", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

BASE = synth_eval_examples(256, seed=256)[:40]


def _code_of(e):
    return dict(zip(e.options, e.slot_codes))


def test_variants_are_the_same_questions_with_one_thing_changed():
    """每个变体逐题对得上原菜单: 同一条消息、同一个正确描述. 256 项的变体保留全部选项, 各自只动它该动的:
    repeat 原样; renumbered_a / _b 换行、仍连续编号; rows_only 换行、每条描述带着原来的码;
    codes_only 行不动、码重新分配. short_random 是正确描述加 59 个随机描述, 按原顺序、连续编号."""
    v = _mod.variants(BASE, seed=0)
    assert set(v) == {"repeat", "renumbered_a", "renumbered_b", "rows_only", "codes_only", "short_random"}
    for name, exs in v.items():
        assert [(e.query, e.label) for e in exs] == [(e.query, e.label) for e in BASE], name
    for b, r in zip(BASE, v["repeat"]):
        assert r == b
    for name in ("renumbered_a", "renumbered_b"):
        assert all(sorted(e.options) == sorted(b.options) and e.codes is None for e, b in zip(v[name], BASE)), name
    assert any(e.options != b.options for e, b in zip(v["renumbered_a"], BASE))
    assert any(a.options != b.options for a, b in zip(v["renumbered_a"], v["renumbered_b"])), "两次打乱不同"
    assert all(_code_of(e) == _code_of(b) and e.options != b.options for e, b in zip(v["rows_only"], BASE))
    assert all(e.options == b.options and e.slot_codes != b.slot_codes for e, b in zip(v["codes_only"], BASE))
    for e, b in zip(v["short_random"], BASE):
        assert len(e.options) == 60 and e.codes is None and b.label in e.options
        assert [c for c in b.options if c in set(e.options)] == e.options, "按原顺序"


def test_variants_depend_only_on_the_seed():
    a, b, c = _mod.variants(BASE, seed=0), _mod.variants(BASE, seed=0), _mod.variants(BASE, seed=1)
    assert a == b and a["codes_only"] != c["codes_only"]


if __name__ == "__main__":
    run(globals())
