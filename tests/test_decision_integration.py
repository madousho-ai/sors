"""Both decision architectures use the public training/checkpoint/serving paths."""

import copy
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from _runner import run
from test_decision import batch, examples, tiny_backbone, tiny_model
from sors.core.checkpoint import load_trained, prepare_from_checkpoint, read_checkpoint, save_trained
from sors.core.decision import architecture_config
from sors.core.menu import with_partners
from sors.evaluation.scoring import EvalSet, evaluate, score_examples
from sors.training.loop import TrainConfig, train


def config(**kw):
    return TrainConfig(**dict(steps=3, batch_size=2, k_max=4, loss="menu", consistency=0.5,
                              micro_batches=2, eval_every=3, log_every=10, save_every=1,
                              lr_lora=1e-3, **kw))


def sampler(n, rng):
    return with_partners([examples()[i % 2] for i in range(n)], rng)


def test_training_evaluation_and_gradient_accumulation_use_the_decision_head():
    for kind in ("minimal", "structural"):
        a, tok, d, _ = tiny_model(kind, True)
        b, _, _, _ = tiny_model(kind, True)
        original = a.blocks[0].read.in_proj_weight.detach().clone()
        states = []
        train(a, tok, d, sampler, {}, config(), on_state=lambda s: states.append(copy.deepcopy(s)))
        train(b, tok, d, sampler, {}, config(accumulate_gradients=True))
        assert not torch.equal(a.blocks[0].read.in_proj_weight, original)
        for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
            torch.testing.assert_close(p, q, atol=3e-5, rtol=3e-4, msg=name)
        scores = evaluate(a, tok, d, EvalSet(examples(), 2), 4, 128, "context-first")
        assert scores["n"] == 2 and scores["top1_in_menu_rate"] == 1
        assert states[-1]["architecture"] == architecture_config(a)


def test_decision_learning_rate_drives_only_the_decision_layers_during_training():
    """lr_decision 只管新增决策层: 主干学习率为 0 时只有决策层在动, 反过来决策层不动;
    第三组的学习率走同一条 warmup/cosine 日程, 完整状态续跑时照样恢复三组."""
    from sors.training.resume import read_state, save_state

    def moved(lr_lora, lr_decision):
        m, tok, d, _ = tiny_model("minimal", True, trainable="full")
        before = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
        cfg = config(lr_decision=lr_decision, lr_schedule="cosine", warmup_steps=1)
        cfg.lr_lora, cfg.lr_embed = lr_lora, 0.0
        train(m, tok, d, sampler, {}, cfg)
        return {n for n, p in m.named_parameters() if p.requires_grad and not torch.equal(p, before[n])}

    head_only = moved(0.0, 1e-3)
    assert head_only and all(not n.startswith("base.") for n in head_only), head_only
    body_only = moved(1e-3, 0.0)
    assert body_only and all(n.startswith("base.") for n in body_only), body_only

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "state.safetensors"
        cfg = config(lr_decision=1e-2, lr_schedule="cosine", warmup_steps=1)
        reference, tok, d, _ = tiny_model("minimal", True, trainable="full")
        train(reference, tok, d, sampler, {}, cfg)
        first, _, _, _ = tiny_model("minimal", True, trainable="full")
        train(first, tok, d, sampler, {}, cfg, stop_after=1, on_state=lambda s: save_state(s, path))
        state = read_state(path)
        assert state["config"]["lr_decision"] == 1e-2
        assert len(state["optimizer"]["param_groups"]) == 3
        resumed, _, _, _ = tiny_model("minimal", True, trainable="full")
        train(resumed, tok, d, sampler, {}, cfg, resume=state)
        for (name, p), (_, q) in zip(reference.named_parameters(), resumed.named_parameters()):
            torch.testing.assert_close(q, p, atol=1e-6, rtol=1e-5, msg=name)


def test_saved_architecture_rebuilds_weights_and_feedback_under_a_new_seed():
    for family in ("qwen3", "qwen35"):
        for kind in ("minimal", "structural"):
            for feedback in (False, True):
                m, tok, d, ids = tiny_model(kind, feedback, family=family)
                with torch.no_grad():
                    for p in m.parameters():
                        if p.requires_grad:
                            p.add_(torch.randn_like(p) * 0.01)
                m.eval()
                data = batch(m, tok, d)
                want = m.forward_batch(data)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "weights.safetensors"
                    save_trained(m, ids, config(), path)
                    record = read_checkpoint(path)
                    assert record.get("architecture") == architecture_config(m), "architecture missing from checkpoint"
                    raw = tiny_backbone(tok, family)
                    torch.manual_seed(992)
                    got, cfg = prepare_from_checkpoint(raw, ids, path)
                    torch.testing.assert_close(got.eval().forward_batch(data), want, atol=0, rtol=0)
                    assert cfg["loss"] == "menu"


def test_checkpoint_rejects_wrong_architecture_and_missing_decision_weights():
    from safetensors import safe_open
    from safetensors.torch import save_file
    m, _, _, ids = tiny_model("structural", True)
    other, _, _, _ = tiny_model("minimal", True)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "weights.safetensors"
        save_trained(m, ids, config(), path)
        try:
            load_trained(other, ids, path)
        except ValueError as exc:
            assert "architecture" in str(exc)
        else:
            raise AssertionError("a structural checkpoint loaded into a minimal model")
        with safe_open(path, framework="pt") as f:
            meta = f.metadata()
            tensors = {k: f.get_tensor(k) for k in f.keys() if k != "blocks.0.read.in_proj_weight"}
        save_file(tensors, path, metadata=meta)
        try:
            load_trained(m, ids, path)
        except ValueError as exc:
            assert "missing" in str(exc)
        else:
            raise AssertionError("a missing decision layer silently kept random weights")


def test_complete_resume_restores_decision_weights_optimizer_and_feedback_mode():
    from sors.training.resume import read_state, save_state
    for kind in ("minimal", "structural"):
        reference, tok, d, _ = tiny_model(kind, True)
        train(reference, tok, d, sampler, {}, config())
        first, _, _, _ = tiny_model(kind, True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.safetensors"
            train(first, tok, d, sampler, {}, config(), stop_after=1, on_state=lambda s: save_state(s, path))
            state = read_state(path)
            resumed, _, _, _ = tiny_model(kind, True)
            train(resumed, tok, d, sampler, {}, config(), resume=state)
            for (name, p), (_, q) in zip(reference.named_parameters(), resumed.named_parameters()):
                torch.testing.assert_close(p, q, atol=0, rtol=0, msg=name)
            wrong, _, _, _ = tiny_model(kind, False)
            try:
                train(wrong, tok, d, sampler, {}, config(), resume=state)
            except ValueError as exc:
                assert "architecture" in str(exc)
            else:
                raise AssertionError("resume accepted a different feedback configuration")


def test_decision_service_bypasses_lm_cache_and_matches_public_scorer():
    from sors.serve.api import SystemOneRequest
    from sors.serve.engine import Engine, RequestTooLong
    from sors.serve.menus import to_example
    questions = SystemOneRequest.model_validate({"model": "m", "state": "red context", "questions": {
        "color": {"type": "choice", "instructions": "Which color?", "criteria": {"red": None, "blue": None}},
        "other": {"type": "choice", "instructions": "Which color?", "criteria": {"green": None, "blue": None}},
    }}).questions
    for kind in ("minimal", "structural"):
        m, tok, d, _ = tiny_model(kind, True)
        exs = [to_example(q, "red context", "") for q in questions.values()]
        expected = score_examples(m, tok, d, exs, 2, 4, 128, "context-first", False)["q"]
        engine = Engine(m, tok, d, max_tokens=128, max_batch_tokens=32)
        with patch("sors.serve.engine.prefix_cache", side_effect=AssertionError("LM cache used")), \
             patch("sors.serve.engine.branch_logits", side_effect=AssertionError("LM cache used")):
            got = engine.evaluate("red context", questions)
        assert list(got.probs) == list(questions)
        assert got.input_tokens > 0
        for row, value in enumerate(got.probs.values()):
            torch.testing.assert_close(torch.tensor(value), torch.tensor(expected[row][:2]), atol=2e-6, rtol=2e-5)
        tiny = Engine(m, tok, d, max_tokens=3)
        with patch.object(m, "forward_batch", side_effect=AssertionError("overlong request reached model")):
            try:
                tiny.evaluate("red context", questions)
            except RequestTooLong:
                pass
            else:
                raise AssertionError("server truncated a decision request")


def train_cli():
    path = Path(__file__).resolve().parents[1] / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("decision_train_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def test_cli_selects_architecture_and_feedback_and_inherits_checkpoint_configuration():
    cli = train_cli()
    assert hasattr(cli, "resolve_architecture"), "architecture CLI is missing"
    for kind in ("minimal", "structural"):
        args = cli.build_parser().parse_args(["--architecture", kind, "--decision-feedback", "--loss", "menu"])
        cfg = cli.resolve_architecture(args)
        assert cfg["kind"] == kind and cfg["feedback"] is True
        assert cli.resolve_adapter(args)["trainable"] == "decision-only"
        m, _, _, ids = tiny_model(kind, True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.safetensors"
            save_trained(m, ids, config(), path)
            args = cli.build_parser().parse_args(["--init", str(path)])
            assert cli.resolve_architecture(args) == architecture_config(m)
            args = cli.build_parser().parse_args(["--init", str(path), "--no-decision-feedback"])
            try:
                cli.resolve_architecture(args)
            except SystemExit:
                pass
            else:
                raise AssertionError("CLI silently changed a checkpoint's feedback mode")


def test_real_train_cli_exports_reloadable_decision_and_complete_state():
    from sors.core.attention import load_causal_lm
    from sors.serve.engine import load_engine
    from sors.training.resume import read_state
    cli = train_cli()
    assert any(a.dest == "save_training_state" for a in cli.build_parser()._actions), "complete-state CLI wiring is missing"
    for kind in ("minimal", "structural"):
        m, tok, d, ids = tiny_model(kind, True)
        with tempfile.TemporaryDirectory() as directory:
            base, out = Path(directory) / "base", Path(directory) / "run"
            tiny_backbone(tok).save_pretrained(base)
            tok.save_pretrained(base)
            argv = ["train.py", "--model", str(base), "--out", str(out), "--dataset", "synth-v5.3", "--architecture", kind,
                    "--decision-feedback", "--decision-dim", "16", "--decision-heads", "4",
                    "--loss", "menu", "--steps", "1", "--batch-size", "2", "--micro-batches", "2",
                    "--save-training-state", "--consistency", "1", "--grad-ckpt", "--temp-max", "999",
                    "--lr-decision", "2e-3"]
            def load(path, **kw):
                return load_causal_lm(path, device="cpu", dtype=torch.float32, **kw)
            with patch("sys.argv", argv), patch.object(cli, "load_causal_lm", side_effect=load), \
                 patch.object(cli, "build_data", return_value=(sampler, {"small": EvalSet(examples(), 2)}, {})):
                cli.main()
            ck = read_checkpoint(out / "trained.safetensors")
            state = read_state(out / "checkpoints/latest.trainstate.safetensors")
            assert state["step"] == 1 and "optimizer" in state
            assert state["config"]["lr_decision"] == 2e-3 and len(state["optimizer"]["param_groups"]) == 3
            assert state["metadata"]["architecture"] == ck["architecture"] == state["architecture"]
            from sors.data.paths import datasets_root
            from sors.training.resume import verify_sampling_fingerprint
            assert state["metadata"]["args"]["datasets_dir"] == str(datasets_root())
            provenance = state["metadata"]["sampling_provenance"]
            assert provenance["datasets_dir"] == str(datasets_root())
            assert len(provenance["code_commit"]) == len(provenance["data_commit"]) == 40
            verify_sampling_fingerprint(Path(__file__).resolve().parents[1], datasets_root(),
                                        state["metadata"]["sampling_fingerprint"])
            assert ck["architecture"]["feedback"] is True
            result = json.loads((out / "result.json").read_text())
            assert result["args"]["architecture"] == kind
            assert result["history"][-1]["eval"]["small"]["n"] == 2
            engine = load_engine(out / "trained.safetensors", base, device="cpu", dtype=torch.float32)
            assert architecture_config(engine.lm) == ck["architecture"]


def test_resume_cli_reconstructs_the_saved_architecture_before_restoring_optimizer():
    path = Path(__file__).resolve().parents[1] / "scripts/resume.py"
    spec = importlib.util.spec_from_file_location("decision_resume_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert hasattr(cli, "prepare_resume_model"), "resume CLI does not reconstruct decision models"
    for kind in ("minimal", "structural"):
        m, tok, _, ids = tiny_model(kind, True)
        args = argparse.Namespace(lora_r=None, lora_alpha=None, lora_dropout=0., trainable="decision-only", grad_ckpt=True)
        record = architecture_config(m)
        state = {"architecture": record, "metadata": {"architecture": record, "adapter": m.adapter}}
        got = cli.prepare_resume_model(tiny_backbone(tok), ids, args, state, None, full=True)
        assert architecture_config(got) == record and got.checkpoint_forward
        state["metadata"]["architecture"] = {**record, "feedback": False}
        try:
            cli.prepare_resume_model(tiny_backbone(tok), ids, args, state, None, full=True)
        except ValueError as exc:
            assert "architecture" in str(exc)
        else:
            raise AssertionError("resume accepted conflicting architecture metadata")


def test_complete_state_save_rejects_datasets_outside_the_resume_contract_before_side_effects():
    cli = train_cli()
    for dataset in ("synth", "banking77", "synth-v5.3+massive"):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "run"
            argv = ["train.py", "--dataset", dataset, "--save-training-state", "--out", str(out)]
            with patch("sys.argv", argv), patch.object(cli, "build_data", side_effect=AssertionError("data loaded before validation")):
                try:
                    cli.main()
                except SystemExit as exc:
                    assert "--save-training-state" in str(exc)
                else:
                    raise AssertionError("created a full state the recovery entry point refuses")
            assert not out.exists()


if __name__ == "__main__":
    run(globals())
