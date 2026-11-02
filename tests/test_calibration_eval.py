"""Calibration reaches both evaluation paths using the existing T=1 forward results."""

import pytest
import torch

from test_evaluate import _tiny
from sors.core.menu import MenuExample, reorder_menu
from sors.core.prompt import DEFAULT_LAYOUT
from sors.evaluation.scoring import EvalSet, consistency_eval, evaluate
from sors.training.loop import TrainConfig, eval_record


def _fixed_logits(monkeypatch, d_ids, rows):
    """Replace the expensive model forward; retain real collation and probability scoring."""
    pending = iter(rows)
    calls = []

    def forward(model, input_ids, attention_mask):
        calls.append(len(input_ids))
        logits = torch.full((len(input_ids), model.config.vocab_size), -torch.inf, dtype=torch.float64)
        for i in range(len(input_ids)):
            q = next(pending)
            logits[i, d_ids[:len(q)]] = torch.tensor(q, dtype=torch.float64).log()
        return logits

    monkeypatch.setattr("sors.evaluation.scoring.last_logits", forward)
    return calls


def test_evaluate_carries_calibration_for_each_type_in_a_mixed_eval_set(monkeypatch):
    tok, ids, model = _tiny()
    examples = [
        MenuExample("a", [10, 11], 0, 10, ["a", "b"]),
        MenuExample("b", [7, 3], 1, 3, ["yes: allowed", "no: refused"], qtype="bool"),
        MenuExample("c", [2, 0, 1], 2, 1, ["high", "low", "middle"], qtype="score"),
    ]
    calls = _fixed_logits(monkeypatch, ids, [[0.81, 0.19], [0.2, 0.8], [0.7, 0.1, 0.2]])
    got = evaluate(model, tok, ids, EvalSet(examples, batch_size=2), 3, 512, DEFAULT_LAYOUT)
    assert got["calib"]["conf_mean"] == pytest.approx(0.77)
    assert got["calib"]["accuracy_minus_confidence"] == pytest.approx(2 / 3 - 0.77)
    assert got["calib"]["choice"]["ece"] == pytest.approx(0.19)
    assert got["calib"]["noul"]["decisive_rate"] == 1.0
    assert got["calib"]["noul"]["brier"] == pytest.approx(0.04)
    assert got["calib"]["score"]["nrps"] == pytest.approx(0.25)
    assert got["accuracy"] == pytest.approx(2 / 3)
    assert got["brier"] == pytest.approx(1.2922 / 3)
    assert calls == [2, 1]


def test_evaluate_uses_the_explicit_positive_class_and_retains_binary_metrics(monkeypatch):
    tok, ids, model = _tiny()
    ex = MenuExample("a", [7, 3], 1, 3, ["positive", "negative"], qtype="bool")
    _fixed_logits(monkeypatch, ids, [[0.2, 0.8]])
    got = evaluate(model, tok, ids, EvalSet([ex], pos_class=7), 2, 512, DEFAULT_LAYOUT)
    assert got["calib"]["noul"]["brier"] == pytest.approx(0.04)
    assert got["brier_binary"] == pytest.approx(0.04)
    assert got["pos_rate"] == 0.0


def test_consistency_calibration_pools_raw_passes_instead_of_averaging_probabilities(monkeypatch):
    tok, ids, model = _tiny()
    ex = MenuExample("a", [0, 1], 0, 0, ["a", "b"])
    passes = [[ex], [reorder_menu(ex, [1, 0])]]
    calls = _fixed_logits(monkeypatch, ids, [[0.81, 0.19], [0.89, 0.11]])
    got = consistency_eval(model, tok, ids, passes, 1, 2, 512, DEFAULT_LAYOUT)
    assert got["calib"]["choice"]["ece"] == pytest.approx(0.35)
    assert got["calib"]["conf_mean"] == pytest.approx(0.85)
    assert got["accuracy"] == 0.5
    assert got["agree"] == 0.0
    assert got["n"] == 1 and got["passes"] == 2
    assert calls == [1, 1]


def test_consistency_calibration_uses_the_eval_sets_explicit_positive_class(monkeypatch):
    tok, ids, model = _tiny()
    ex = MenuExample("a", [7, 3], 1, 3, ["positive", "negative"], qtype="bool")
    calls = _fixed_logits(monkeypatch, ids, [[0.2, 0.8]])
    got = consistency_eval(model, tok, ids, [[ex]], 1, 2, 512, DEFAULT_LAYOUT, pos_class=7)
    assert got["calib"]["noul"]["brier"] == pytest.approx(0.04)
    assert got["calib"]["noul"]["decisive_accuracy"] == 1.0
    assert calls == [1]


def test_periodic_eval_reuses_probe_forwards_and_preserves_the_positive_class(monkeypatch):
    tok, ids, model = _tiny()
    ex = MenuExample("a", [7, 3], 1, 3, ["positive", "negative"], qtype="bool")
    calls = _fixed_logits(monkeypatch, ids, [[0.2, 0.8]])
    got = eval_record(model, tok, ids, {"custom": EvalSet([ex], pos_class=7)}, {"custom": [[ex]]},
                      TrainConfig(k_max=2), final=False)
    assert got["consistency"]["custom"]["calib"]["noul"]["brier"] == pytest.approx(0.04)
    assert calls == [1]
