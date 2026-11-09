"""--cot-passes N: 按轮抽时, 带计算过程 (cot) 的绑定每轮出 N 遍, 覆盖它所在目标的遍数; 其余绑定不变."""

import random
from dataclasses import replace

import pytest

from test_synth_v5 import PAIR_GOALS, _paired
from sors.data.synth_v5 import V5Rounds, parse_passes


def _items():
    items = _paired()
    return [replace(it, cot="One. Two.") if i == 0 else it for i, it in enumerate(items)]


def test_cot_bindings_appear_n_times_per_round_and_the_others_keep_their_passes():
    items = _items()
    plain = V5Rounds(items, parse_passes(None, PAIR_GOALS))
    boosted = V5Rounds(items, parse_passes(None, PAIR_GOALS), cot_passes=5)
    count = lambda s: sum(items[u[0][0]].cot is not None for u in s.units)
    assert count(plain) == 1 and count(boosted) == 5
    assert len(boosted.units) - count(boosted) == len(plain.units) - count(plain)
    out = boosted(boosted.round_size, random.Random(0))
    assert sum(ex.cot is not None for ex in out) == 5


def test_cot_passes_must_be_a_positive_whole_number():
    with pytest.raises(ValueError, match="cot_passes"):
        V5Rounds(_items(), parse_passes(None, PAIR_GOALS), cot_passes=0)


def test_cli_passes_cot_passes_to_the_round_sampler():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("train_cli", Path(__file__).parents[1] / "scripts/train.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    args = cli.build_parser().parse_args(["--passes", "complex=2", "--cot-passes", "8"])
    assert args.cot_passes == 8
