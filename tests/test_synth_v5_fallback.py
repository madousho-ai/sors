"""Exercise fallback augmentation through the real loader and samplers; no model needed."""

import copy
import random

from _runner import run
from test_synth_v5 import TELECOM_C, TELECOM_Q, _data, _find, _paired, PAIR_GOALS
from decidophobia.core.menu import row_alignment, with_partners
from decidophobia.data.synth_v5 import V5Rounds, V5Sampler, item_example, load_synth_v5, parse_mix, parse_passes


def _marked(reason="insufficient_evidence", fallback="unknown", control=False):
    contexts = copy.deepcopy(TELECOM_C)
    binding = contexts[0]["questions"][2]
    binding["fallback"] = fallback
    if not control:
        binding.update(answer={"no": 1, "yes": 1}, uncertainty=reason, why="The required fact is absent.")
    return _find(load_synth_v5(_data(telecom=(TELECOM_Q, contexts))),
                 "telecom_cancel_contract#0", "detail_cancel_contract")


def test_unknown_augmentation_changes_the_target_and_bool_type_then_shuffles():
    item = _marked()
    rng = random.Random(12)
    positions = set()
    for _ in range(30):
        ex = item_example(item, rng, 0.0, fallback_rate=1.0)
        assert len(ex.options) == 3 and ex.qtype == "choice"
        assert ex.target is None and "Insufficient information" in ex.option_names[ex.gold_idx]
        positions.add(ex.gold_idx)
        a, b = with_partners([ex], rng)
        assert sorted(a.option_names) == sorted(b.option_names)
        assert row_alignment(a, b)[a.gold_idx] == b.gold_idx
    assert positions == {0, 1, 2}
    assert item.target == [0.5, 0.5] and item.rows == ["no", "yes"]


def test_disabling_fallback_preserves_the_original_uniform_bool_menu():
    ex = item_example(_marked(), random.Random(1), 0.0, fallback_rate=0.0)
    assert ex.qtype == "bool" and sorted(ex.option_names) == ["no", "yes"] and ex.target == [0.5, 0.5]


def test_menu_missing_uses_other_and_a_clear_control_keeps_its_answer():
    positive = item_example(_marked("menu_missing", "other"), random.Random(0), 0.0, fallback_rate=1.0)
    assert "None of these" in positive.option_names[positive.gold_idx]
    for kind in ("other", "unknown"):
        ex = item_example(_marked(fallback=kind, control=True), random.Random(0), 0.0, fallback_rate=1.0)
        assert len(ex.options) == 3 and ex.target is None and ex.option_names[ex.gold_idx] == "no"


def test_default_fallback_rate_produces_both_views_about_equally():
    item, rng = _marked(), random.Random(0)
    added = sum(len(item_example(item, rng, 0.0).options) == 3 for _ in range(2000))
    assert 900 < added < 1100, added


def test_existing_other_pairs_emit_one_view_per_draw_in_both_samplers():
    items = _paired()
    for rate in (0.0, 0.5, 1.0):
        samplers = [V5Sampler(items, parse_mix(None, PAIR_GOALS), fallback_rate=rate),
                    V5Rounds(items, parse_passes(None, PAIR_GOALS), fallback_rate=rate)]
        for sampler in samplers:
            rng = random.Random(4)
            queue_sizes = set()
            for _ in range(100):
                batch = sampler(8, rng)
                assert len(batch) == 8
                for ex in batch:
                    if "floor 2 printer" in ex.query or "today: why" in ex.query:
                        queue_sizes.add(len(ex.options))
                        if len(ex.options) == 4:
                            assert ex.option_names[ex.gold_idx] == "None of these alerts"
                        else:
                            assert ex.target == [1 / 3] * 3
            assert queue_sizes == ({3} if rate == 0 else {4} if rate == 1 else {3, 4}), queue_sizes


def test_invalid_fallback_rates_are_rejected_before_sampling():
    items = _paired()
    for rate in (-0.1, 1.1, float("nan"), float("inf")):
        for construct in (lambda: V5Sampler(items, parse_mix(None, PAIR_GOALS), fallback_rate=rate),
                          lambda: V5Rounds(items, parse_passes(None, PAIR_GOALS), fallback_rate=rate)):
            try:
                construct()
            except ValueError:
                continue
            raise AssertionError(f"accepted fallback rate {rate}")


def test_a_masked_keyed_control_preserves_the_fallback_meaning_and_soft_weights():
    contexts = copy.deepcopy(TELECOM_C)
    contexts[0]["questions"][0].update(answer={"cancel_contract": 3, "port_number_out": 1}, fallback="other")
    item = _find(load_synth_v5(_data(telecom=(TELECOM_Q, contexts))), "telecom_cancel_contract#0", "intent")
    ex = item_example(item, random.Random(0), mask_rate=1.0, fallback_rate=1.0)
    assert dict(zip(ex.option_names, ex.target)) == {
        "cancel_contract": 0.75, "port_number_out": 0.25, "view_latest_bill": 0.0,
        "3g_switch_off": 0.0, "none_of_these": 0.0}
    assert ex.option_names[ex.gold_idx] == "cancel_contract"


def test_partial_unknown_score_distribution_becomes_an_explicit_unknown_choice():
    contexts = copy.deepcopy(TELECOM_C)
    contexts[3]["questions"][0].update(uncertainty="insufficient_evidence", fallback="unknown",
                                       why="The severity could be either of the upper two levels.")
    item = _find(load_synth_v5(_data(telecom=(TELECOM_Q, contexts))), "telecom_rule_1", "level")
    ex = item_example(item, random.Random(0), 0.0, fallback_rate=1.0)
    assert len(ex.options) == 4 and ex.qtype == "choice" and ex.target is None
    assert ex.option_names[ex.gold_idx] == "Insufficient information to determine the answer"
    assert item.target == [0.0, 0.25, 0.75]


if __name__ == "__main__":
    run(globals())
