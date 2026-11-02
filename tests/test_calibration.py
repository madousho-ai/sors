"""Hand-computed T=1 calibration examples, in menu-position and ordered-class space."""

import pytest

from sors.core.menu import MenuExample
from sors.evaluation import metrics


def test_choice_calibration_pools_top_labels_in_ten_equal_width_bins():
    # Both confidences land in [0.8, 0.9): |0.85 - 0.5| = 0.35.
    got = metrics.choice_calibration([[0.81, 0.19], [0.89, 0.11]], [0, 1])
    assert got == pytest.approx({"ece": 0.35, "calibration_score": 30.0,
                                 "conf_mean": 0.85, "accuracy_minus_confidence": -0.35,
                                 "conf_correct_mean": 0.81, "conf_wrong_mean": 0.89})


def test_confidence_gap_reports_underconfidence_and_missing_conditionals():
    got = metrics.confidence_summary([[0.6, 0.4], [0.3, 0.7]], [0, 1])
    assert got["conf_mean"] == pytest.approx(0.65)
    assert got["accuracy_minus_confidence"] == pytest.approx(0.35)
    assert got["conf_correct_mean"] == pytest.approx(0.65)
    assert got["conf_wrong_mean"] is None
    wrong = metrics.choice_calibration([[1.0, 0.0]], [1])
    assert wrong["conf_correct_mean"] is None
    assert wrong["conf_wrong_mean"] == 1.0
    assert wrong["calibration_score"] == 0.0
    assert metrics.choice_calibration([[1.0, 0.0]], [0])["calibration_score"] == 100.0


def test_noul_decisive_band_includes_both_edges_and_counts_abstentions_wrong():
    # Three decisive, two correct; one correct abstention. Acc (abstentions wrong) = 2/4.
    got = metrics.noul_calibration([0.8, 0.2, 0.9, 0.5], [1, 0, 0, 1])
    assert got["decisive_rate"] == 0.75
    assert got["decisive_accuracy"] == pytest.approx(2 / 3)
    assert got["accuracy_with_abstentions_wrong"] == 0.5
    assert got["competence"] == 0.0
    assert got["ece"] == pytest.approx(0.45)
    assert got["brier"] == pytest.approx(0.285)
    assert got["calibration_score"] == pytest.approx(10.0)
    assert got["conf_mean"] == pytest.approx(0.75)
    assert got["accuracy_minus_confidence"] == 0.0
    assert got["conf_correct_mean"] == pytest.approx(0.7)
    assert got["conf_wrong_mean"] == 0.9


def test_noul_all_abstentions_and_chance_corrected_competence():
    abstain = metrics.noul_calibration([0.200001, 0.799999], [0, 1])
    assert abstain["decisive_rate"] == 0.0
    assert abstain["decisive_accuracy"] is None
    assert abstain["accuracy_with_abstentions_wrong"] == 0.0
    assert abstain["competence"] == 0.0
    assert metrics.noul_calibration([0.9] * 4, [1, 1, 1, 0])["competence"] == 50.0
    assert metrics.noul_calibration([0.0, 1.0], [0, 1])["competence"] == 100.0


def test_score_calibration_uses_ordered_levels_and_normalizes_each_menu_width():
    # K=3, gold 1: ((.1-0)^2 + (.3-1)^2)/2 = .25; central interval [0,2].
    # K=2, gold 1: (.96-0)^2 = .9216; central interval [0,0], misses gold.
    got = metrics.score_calibration([[0.1, 0.2, 0.7], [0.96, 0.04]], [1, 1])
    assert got["nrps"] == pytest.approx(0.5858)
    assert got["interval90_coverage"] == 0.5
    assert got["interval90_width"] == 0.5
    assert got["ece"] == pytest.approx(0.83)


def test_score_interval_inverts_the_cdf_at_exact_five_and_ninety_five_percent():
    got = metrics.score_calibration([[0.05, 0.90, 0.05]], [2])
    assert got["interval90_coverage"] == 0.0
    assert got["interval90_width"] == 0.5


def test_calibration_summary_maps_permuted_positions_and_splits_question_types():
    examples = [
        MenuExample("a", [10, 11], 0, 10, ["a", "b"]),
        MenuExample("b", [7, 3], 1, 3, ["yes: allowed", "no: refused"], qtype="bool"),
        MenuExample("c", [2, 0, 1], 2, 1, ["high", "low", "middle"], qtype="score"),
    ]
    got = metrics.calibration_summary([[0.8, 0.2, 0.0], [0.2, 0.8, 0.0], [0.7, 0.1, 0.2]], examples,
                                      pos_class=7)
    assert got["conf_mean"] == pytest.approx(2.3 / 3)
    assert got["accuracy_minus_confidence"] == pytest.approx(-0.1)
    assert got["choice"]["ece"] == pytest.approx(0.2)
    assert got["noul"]["decisive_rate"] == 1.0
    assert got["noul"]["brier"] == pytest.approx(0.04)
    assert got["score"]["nrps"] == pytest.approx(0.25)
    assert got["score"]["interval90_width"] == 1.0
    inferred = metrics.calibration_summary([[0.2, 0.8]], examples[1:2])
    assert inferred["noul"]["brier"] == pytest.approx(0.04)


def test_calibration_summary_handles_empty_input():
    assert metrics.calibration_summary([], []) == {}
