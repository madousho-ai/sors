"""Shared prefix execution is selectable and preserved across public interfaces."""

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from _runner import run
from test_candidate import candidate_batch, candidate_model, examples, tiny_backbone
from test_candidate_integration import requests
from test_candidate_prefix import set_mode, trace
from test_decision_integration import train_cli
from sors.core.checkpoint import prepare_from_checkpoint, read_checkpoint, save_trained
from sors.training.loop import TrainConfig, train


def test_train_config_applies_auto_mode_and_checkpointing_does_not_reencode_prefixes():
    assert "candidate_prefix_cache" in TrainConfig.__dataclass_fields__, "prefix execution is absent from training config"
    m, tok, d, _ = candidate_model(1, checkpointed=True)
    cfg = TrainConfig(steps=2, batch_size=1, k_max=4, loss="menu", log_every=10, eval_every=10,
                      candidate_prefix_cache="auto")
    with trace(m) as calls:
        train(m, tok, d, lambda n, rng: [examples()[0]] * n, {}, cfg)
    assert len([c for c in calls if c["cache"] and c["past"] == 0]) == 2


def test_weight_checkpoint_preserves_execution_mode_and_old_candidate_defaults_to_off():
    from safetensors import safe_open
    from safetensors.torch import save_file
    for mode in ("auto", "on", "off"):
        m, tok, _, ids = candidate_model(0)
        set_mode(m, mode)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.safetensors"
            save_trained(m, ids, TrainConfig(loss="menu"), path)
            cfg = read_checkpoint(path)["config"]
            assert cfg.get("candidate_prefix_cache") == mode, "actual model execution mode was not saved"
            restored, _ = prepare_from_checkpoint(tiny_backbone(tok), ids, path)
            assert restored.candidate_prefix_cache == mode
            with safe_open(path, framework="pt") as f:
                metadata = f.metadata()
                tensors = {k: f.get_tensor(k) for k in f.keys()}
            old_cfg = json.loads(metadata["config"])
            del old_cfg["candidate_prefix_cache"]
            metadata["config"] = json.dumps(old_cfg)
            save_file(tensors, path, metadata=metadata)
            old, _ = prepare_from_checkpoint(tiny_backbone(tok), ids, path)
            assert old.candidate_prefix_cache == "off"


def test_training_and_serving_clis_expose_prefix_mode():
    cli = train_cli()
    assert any(a.dest == "candidate_prefix_cache" for a in cli.build_parser()._actions), "training cache flag is missing"
    for mode in ("auto", "on", "off"):
        args = cli.build_parser().parse_args(["--architecture", "candidate", "--candidate-prefix-cache", mode])
        assert cli.resolve_candidate_prefix_cache(args, None) == mode
    args = cli.build_parser().parse_args(["--architecture", "candidate"])
    assert cli.resolve_candidate_prefix_cache(args, None) == "auto"
    args.init = "already-read-checkpoint"
    assert cli.resolve_candidate_prefix_cache(args, {}) == "off"
    assert cli.resolve_candidate_prefix_cache(args, {"candidate_prefix_cache": "on"}) == "on"
    path = Path(__file__).resolve().parents[1] / "scripts/serve.py"
    spec = importlib.util.spec_from_file_location("prefix_serve_cli", path)
    serve = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(serve)
    assert any(a.dest == "candidate_prefix_cache" for a in serve.build_parser()._actions), "serving cache flag is missing"
    assert serve.build_parser().parse_args(["--init", "x", "--candidate-prefix-cache", "off"]).candidate_prefix_cache == "off"


def test_service_counts_actual_shared_encoding_and_allows_full_reference_override():
    from sors.serve.engine import Engine
    m, tok, d, _ = candidate_model(0)
    qs = requests()
    set_mode(m, "auto")
    with trace(m) as calls:
        result = Engine(m, tok, d, max_tokens=128, max_batch_tokens=32).evaluate("red context", qs)
    assert result.input_tokens == sum(c["ids"].numel() for c in calls)
    assert sum(c["cache"] and c["past"] == 0 for c in calls) == len(qs)
    with trace(m) as calls:
        full = Engine(m, tok, d, max_tokens=128, candidate_prefix_cache="off").evaluate("red context", qs)
    assert all(not c["cache"] for c in calls)
    assert result.input_tokens < full.input_tokens
    for key in qs:
        torch.testing.assert_close(torch.tensor(result.probs[key]), torch.tensor(full.probs[key]), atol=3e-5, rtol=3e-4)


def test_complete_resume_uses_the_recorded_prefix_mode():
    from sors.training.resume import read_state, save_state
    assert "candidate_prefix_cache" in TrainConfig.__dataclass_fields__, "prefix execution is absent from training config"
    m, tok, d, ids = candidate_model(0)
    cfg = TrainConfig(steps=2, batch_size=1, k_max=4, loss="menu", log_every=10, eval_every=10,
                      candidate_prefix_cache="on", lr_lora=1e-3)
    sample = lambda n, rng: [examples()[0]] * n
    train(m, tok, d, sample, {}, cfg)
    first, _, _, _ = candidate_model(0)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "state.safetensors"
        train(first, tok, d, sample, {}, cfg, stop_after=1, on_state=lambda s: save_state(s, path))
        state = read_state(path)
        assert state["config"]["candidate_prefix_cache"] == "on"
        resumed, _, _, _ = candidate_model(0)
        set_mode(resumed, "off")
        with trace(resumed) as calls:
            train(resumed, tok, d, sample, {}, cfg, resume=state)
        assert resumed.candidate_prefix_cache == "on"
        assert len([c for c in calls if c["cache"] and c["past"] == 0]) == 1
        for p, q in zip(m.parameters(), resumed.parameters()):
            torch.testing.assert_close(p, q, atol=0, rtol=0)


def test_service_cache_budget_reserves_the_original_prefix_and_fork_workspace():
    from sors.serve.engine import Engine
    from sors.serve.api import SystemOneRequest
    qs = SystemOneRequest.model_validate({"model": "m", "state": "red context", "questions": {
        "pick": {"type": "choice", "instructions": "Which color?",
                 "criteria": {f"color{i}": None for i in range(8)}}}}).questions
    for state, budget in (("red context", 24), ("red context", 48), ("red context " * 100, 840)):
        m, tok, d, _ = candidate_model(0)
        with trace(m) as calls:
            Engine(m, tok, d, max_tokens=1024, max_batch_tokens=budget).evaluate(state, qs)
        for call in calls:
            if call["past"]:
                count, suffix = call["ids"].shape
                prefix = call["past"]
                assert prefix + count * (prefix + suffix) <= budget
                if count > 1:
                    assert (count + 2) * prefix <= budget, "deepcopy/reorder temporary cache exceeded the budget"


if __name__ == "__main__":
    run(globals())
