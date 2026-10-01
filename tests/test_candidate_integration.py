"""Candidate CLI, training, persistence and serving use the same joint encoder."""

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from _runner import run
from test_candidate import candidate_batch, candidate_model, examples, joint_text, tiny_backbone
from test_decision_integration import config, sampler, train_cli
from decidophobia.core.checkpoint import load_trained, prepare_from_checkpoint, read_checkpoint, save_trained
from decidophobia.core.decision import architecture_config
from decidophobia.evaluation.scoring import EvalSet, score_examples
from decidophobia.training.loop import TrainConfig, train
from decidophobia.training.resume import read_state, save_state


def test_cli_exposes_candidate_with_zero_one_two_set_blocks_and_frozen_default():
    cli = train_cli()
    action = next(a for a in cli.build_parser()._actions if a.dest == "architecture")
    assert "candidate" in action.choices, "candidate CLI is missing"
    tags = []
    for blocks in (0, 1, 2):
        args = cli.build_parser().parse_args(["--architecture", "candidate", "--decision-blocks", str(blocks), "--loss", "menu"])
        record = cli.resolve_architecture(args)
        assert record["blocks"] == blocks and record["kind"] == "candidate"
        assert cli.resolve_adapter(args)["trainable"] == "decision-only"
        tags.append(cli.run_tag(args))
    assert len(set(tags)) == 3 and all("candidate" in tag for tag in tags)
    for flags in (["--decision-feedback"], ["--decision-layers", "0,1"]):
        args = cli.build_parser().parse_args(["--architecture", "candidate", *flags])
        try:
            cli.resolve_architecture(args)
        except SystemExit:
            pass
        else:
            raise AssertionError("candidate silently accepted cross-layer wiring")


def test_candidate_checkpoint_rebuilds_the_exact_set_count_and_scalar_head():
    for family in ("qwen3", "qwen35"):
        for blocks in (0, 1, 2):
            m, tok, d, ids = candidate_model(blocks, family)
            with torch.no_grad():
                for p in m.parameters():
                    if p.requires_grad:
                        p.add_(0.01 * torch.randn_like(p))
            want = m.eval().forward_batch(candidate_batch(tok, d))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "weights.safetensors"
                save_trained(m, ids, config(), path)
                raw = tiny_backbone(tok, family)
                torch.manual_seed(777)
                got, _ = prepare_from_checkpoint(raw, ids, path)
                assert architecture_config(got) == architecture_config(m)
                torch.testing.assert_close(got.eval().forward_batch(candidate_batch(tok, d)), want, atol=0, rtol=0)
                args = train_cli().build_parser().parse_args(["--init", str(path)])
                assert train_cli().resolve_architecture(args) == architecture_config(m)
                other, _, _, _ = candidate_model((blocks + 1) % 3, family)
                try:
                    load_trained(other, ids, path)
                except ValueError as exc:
                    assert "architecture" in str(exc)
                else:
                    raise AssertionError("checkpoint silently changed SetBlock count")


def test_candidate_training_accumulation_and_complete_resume_match():
    for blocks in (0, 2):
        a, tok, d, _ = candidate_model(blocks)
        b, _, _, _ = candidate_model(blocks)
        original = a.scorer.weight.detach().clone()
        train(a, tok, d, sampler, {}, config())
        train(b, tok, d, sampler, {}, config(accumulate_gradients=True))
        assert not torch.equal(a.scorer.weight, original)
        for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
            torch.testing.assert_close(p, q, atol=3e-5, rtol=3e-4, msg=name)
        first, _, _, _ = candidate_model(blocks)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "complete.trainstate.safetensors"
            train(first, tok, d, sampler, {}, config(), stop_after=1, on_state=lambda s: save_state(s, path))
            state = read_state(path)
            resumed, _, _, _ = candidate_model(blocks)
            train(resumed, tok, d, sampler, {}, config(), resume=state)
            for (name, p), (_, q) in zip(a.named_parameters(), resumed.named_parameters()):
                torch.testing.assert_close(p, q, atol=0, rtol=0, msg=name)


def requests():
    from decidophobia.serve.api import SystemOneRequest
    return SystemOneRequest.model_validate({"model": "m", "state": "red context", "questions": {
        "color": {"type": "choice", "instructions": "Which color?", "criteria": {"red": None, "blue": None}},
        "long": {"type": "choice", "instructions": "Which color?", "criteria": {"long green": None, "blue": None}},
    }}).questions


def test_candidate_service_counts_only_joint_branches_and_matches_public_scorer():
    from decidophobia.serve.engine import Engine
    from decidophobia.serve.menus import to_example
    m, tok, d, _ = candidate_model(1)
    m.set_candidate_prefix_cache("off")  # Full-branch accounting remains available as a reference.
    qs = requests()
    exs = [to_example(q, "red context", "") for q in qs.values()]
    want_tokens = sum(len(tok.encode(joint_text(ex, name), add_special_tokens=False))
                      for ex in exs for name in ex.option_names)
    engine = Engine(m, tok, d, max_tokens=128, max_batch_tokens=20)
    with patch("decidophobia.serve.engine.prefix_cache", side_effect=AssertionError("LM prefix cache used")):
        result = engine.evaluate("red context", qs)
    assert result.input_tokens == want_tokens, "logical prefix counted as an extra model forward"
    expected = score_examples(m, tok, d, exs, 2, 4, 128, "context-first", False)["q"]
    for i, p in enumerate(result.probs.values()):
        torch.testing.assert_close(torch.tensor(p), torch.tensor(expected[i][:len(p)]), atol=2e-6, rtol=2e-5)


def test_candidate_preview_shows_the_shared_prefix_and_per_candidate_readout():
    from decidophobia.serve.engine import Engine
    from decidophobia.serve.menus import to_example
    m, tok, d, _ = candidate_model(0)
    qs = requests()
    exs = [to_example(q, "red context", "") for q in qs.values()]
    preview = Engine(m, tok, d).prompts("red context", qs)
    for qid, ex in zip(qs, exs):
        prefix, branches = preview[qid]
        assert "red context" in prefix and "Which color?" in prefix
        assert "Candidate branch 1" in branches and "Answer:" in branches
        assert "<|D" not in prefix + branches


def test_candidate_service_rejects_any_overlong_branch_before_running_any_question():
    from decidophobia.serve.engine import Engine, RequestTooLong
    qs = requests()
    qs["long"].criteria["long green"] = "long context " * 100
    m, tok, d, _ = candidate_model(0)
    with patch.object(m, "forward_batch", side_effect=AssertionError("model ran before whole-request validation")):
        try:
            Engine(m, tok, d, max_tokens=40).evaluate("red context", qs)
        except RequestTooLong:
            pass
        else:
            raise AssertionError("candidate request was truncated")


def test_candidate_actual_cli_exports_complete_state_and_loadable_service():
    from decidophobia.core.attention import load_causal_lm
    from decidophobia.serve.engine import load_engine
    cli = train_cli()
    assert "candidate" in next(a for a in cli.build_parser()._actions if a.dest == "architecture").choices
    for blocks in (0, 2):
        _, tok, d, _ = candidate_model(blocks)
        with tempfile.TemporaryDirectory() as directory:
            base, out = Path(directory) / "base", Path(directory) / "run"
            tiny_backbone(tok).save_pretrained(base)
            tok.save_pretrained(base)
            argv = ["train.py", "--model", str(base), "--out", str(out), "--dataset", "synth-v5.1",
                    "--architecture", "candidate", "--decision-blocks", str(blocks), "--decision-dim", "16",
                    "--loss", "menu", "--steps", "1", "--batch-size", "2", "--consistency", "1",
                    "--save-training-state", "--grad-ckpt", "--temp-max", "999"]
            def load(path, **kw):
                return load_causal_lm(path, device="cpu", dtype=torch.float32, **kw)
            with patch("sys.argv", argv), patch.object(cli, "load_causal_lm", side_effect=load), \
                 patch.object(cli, "build_data", return_value=(sampler, {"small": EvalSet(examples(), 2)}, {})):
                cli.main()
            ck = read_checkpoint(out / "trained.safetensors")
            state = read_state(out / "checkpoints/latest.trainstate.safetensors")
            assert state["architecture"] == state["metadata"]["architecture"] == ck["architecture"]
            assert state["architecture"]["blocks"] == blocks
            result = json.loads((out / "result.json").read_text())
            assert result["args"]["architecture"] == "candidate" and result["args"]["trainable"] == "decision-only"
            engine = load_engine(out / "trained.safetensors", base, device="cpu", dtype=torch.float32)
            assert len(engine.evaluate("red context", requests()).probs) == 2
            script = Path(__file__).resolve().parents[1] / "scripts/resume.py"
            spec = importlib.util.spec_from_file_location("candidate_resume_cli", script)
            resume_cli = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(resume_cli)
            restored = resume_cli.prepare_resume_model(tiny_backbone(tok), state["metadata"]["train_ids"],
                                                       argparse.Namespace(**state["metadata"]["args"]),
                                                       state, None, full=True)
            train(restored, tok, d, sampler, {}, TrainConfig(**state["config"]), resume=state)
            data = candidate_batch(tok, d)
            torch.testing.assert_close(restored.eval().forward_batch(data), engine.lm.forward_batch(data), atol=0, rtol=0)


if __name__ == "__main__":
    run(globals())
