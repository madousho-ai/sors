"""Exercise threaded preparation through training/resume CLI configuration."""

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
import threading
from unittest.mock import patch

from _runner import run
from test_decision import tokenizer, tiny_backbone
from test_resume import _Sampler
from sors.training import loop


def _cli(name):
    spec = importlib.util.spec_from_file_location(f"prefetch_{name}_cli", Path(__file__).parents[1] / f"scripts/{name}.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def _parse(cli, flags):
    try:
        return cli.build_parser().parse_args(flags)
    except SystemExit:
        raise AssertionError("data preparation controls are missing from the CLI") from None


@dataclass
class _CPUAttention:
    implementation: str = "sdpa"
    device: str = "cpu"


def test_train_cli_controls_real_preparation_and_records_it_in_checkpoints():
    """Missing flag-to-TrainConfig wiring changes the observed collate thread."""
    from safetensors import safe_open

    for flags, workers, capacity in ((["--data-backend", "thread"], 2, 2),
                                      (["--data-workers", "0", "--data-prefetch", "1"], 0, 1),
                                      (["--data-backend", "thread", "--data-workers", "1", "--data-prefetch", "3"], 1, 3)):
        cli = _cli("train")
        _parse(cli, flags)
        tok, _, _ = tokenizer()
        lm = tiny_backbone(tok)
        calls, original = [], loop.collate

        def observed(*args, **kwargs):
            calls.append(threading.current_thread())
            return original(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "run"
            argv = ["train.py", "--dataset", "synth-v5.1", "--datasets-dir", directory,
                    "--out", str(out), "--steps", "2", "--batch-size", "2", "--k-max", "3", "--k-eval", "3",
                    "--max-length", "128", "--consistency", "0.7", "--eval-every", "2", "--temp-max", "200", *flags]
            with patch.object(sys, "argv", argv), \
                 patch.object(cli, "build_data", return_value=(_Sampler(), {}, {})), \
                 patch.object(cli.AutoTokenizer, "from_pretrained", return_value=tok), \
                 patch.object(cli, "load_causal_lm", return_value=(lm, _CPUAttention())), \
                 patch.object(loop, "collate", observed):
                cli.main()
            with safe_open(out / "trained.safetensors", framework="pt") as checkpoint:
                saved = json.loads(checkpoint.metadata()["config"])
            assert saved.get("data_workers") == workers and saved.get("data_prefetch") == capacity
            args = json.loads((out / "result.json").read_text())["args"]
            assert (args.get("data_workers"), args.get("data_prefetch")) == (workers, capacity)
        assert len(calls) == 2
        assert all((t is threading.current_thread()) == (workers == 0) for t in calls)
        assert all(t is threading.current_thread() or not t.is_alive() for t in calls)


def test_train_cli_defaults_to_process_workers_and_records_gpu_prefetch():
    cli = _cli("train")
    args = _parse(cli, [])
    assert (args.data_backend, args.device_prefetch) == ("process", 1)
    args = _parse(cli, ["--data-backend", "thread", "--device-prefetch", "3"])
    assert (args.data_backend, args.device_prefetch) == ("thread", 3)
    for flags in (["--data-backend", "fibers"],):
        try:
            cli.build_parser().parse_args(flags)
        except SystemExit:
            continue
        raise AssertionError(f"{flags} accepted")


def test_train_cli_records_token_budget_grouping_and_per_group_backward():
    """Missing flag-to-TrainConfig wiring leaves the saved config at its defaults."""
    from safetensors import safe_open

    cli = _cli("train")
    args = _parse(cli, [])
    assert (args.micro_tokens, args.accumulate_gradients) == (0, False)
    tok, _, _ = tokenizer()
    lm = tiny_backbone(tok)
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "run"
        argv = ["train.py", "--dataset", "synth-v5.1", "--datasets-dir", directory,
                "--out", str(out), "--steps", "2", "--batch-size", "2", "--k-max", "3", "--k-eval", "3",
                "--max-length", "128", "--consistency", "0.7", "--eval-every", "2", "--temp-max", "200",
                "--data-workers", "0", "--micro-tokens", "300", "--accumulate-gradients"]
        with patch.object(sys, "argv", argv), \
             patch.object(cli, "build_data", return_value=(_Sampler(), {}, {})), \
             patch.object(cli.AutoTokenizer, "from_pretrained", return_value=tok), \
             patch.object(cli, "load_causal_lm", return_value=(lm, _CPUAttention())):
            cli.main()
        with safe_open(out / "trained.safetensors", framework="pt") as checkpoint:
            saved = json.loads(checkpoint.metadata()["config"])
    assert (saved.get("micro_tokens"), saved.get("accumulate_gradients")) == (300, True), saved
    for flags in (["--micro-tokens", "-1"],):
        with patch.object(sys, "argv", ["train.py", *flags]), \
             patch.object(cli, "build_data", side_effect=AssertionError("invalid budget reached data loading")):
            try:
                cli.main()
            except SystemExit as error:
                assert "micro-tokens" in str(error), str(error)
            else:
                raise AssertionError("negative token budget accepted")


def test_resume_overrides_the_token_budget_as_an_execution_setting():
    cli = _cli("resume")
    saved = {"steps": 20, "batch_size": 36, "micro_batches": 8}
    assert cli.resume_config(saved, _parse(cli, ["--run", "runs/example"])).micro_tokens == 0
    cfg = cli.resume_config(saved, _parse(cli, ["--run", "runs/example", "--micro-tokens", "16384"]))
    assert cfg.micro_tokens == 16384
    try:
        cli.resume_config(saved, _parse(cli, ["--run", "runs/example", "--micro-tokens", "-1"]))
    except ValueError:
        pass
    else:
        raise AssertionError("negative token budget accepted")


def test_resume_overrides_backend_and_gpu_prefetch_as_execution_settings():
    cli = _cli("resume")
    saved = {"steps": 20, "batch_size": 36, "micro_batches": 8, "data_backend": "thread", "device_prefetch": 0}
    cfg = cli.resume_config(saved, _parse(cli, ["--run", "runs/example"]))
    assert (cfg.data_backend, cfg.device_prefetch) == ("thread", 0)
    cfg = cli.resume_config(saved, _parse(cli, ["--run", "runs/example", "--data-backend", "process",
                                                "--device-prefetch", "2"]))
    assert (cfg.data_backend, cfg.device_prefetch) == ("process", 2)
    try:
        cli.resume_config(saved, _parse(cli, ["--run", "runs/example", "--device-prefetch", "-1"]))
    except ValueError:
        pass
    else:
        raise AssertionError("negative device prefetch accepted")


def test_training_cli_rejects_invalid_limits_before_data_or_model_loading():
    cli = _cli("train")
    for flags in (["--data-workers", "-1"], ["--data-prefetch", "0"]):
        _parse(cli, flags)
        with patch.object(sys, "argv", ["train.py", *flags]), \
             patch.object(cli, "build_data", side_effect=AssertionError("invalid limits reached data loading")):
            try:
                cli.main()
            except SystemExit as error:
                assert "data_" in str(error), str(error)
            else:
                raise AssertionError("invalid preparation settings accepted")


def test_resume_controls_preserve_saved_limits_and_override_only_execution_settings():
    cli = _cli("resume")
    assert hasattr(cli, "resume_config"), "resume configuration does not support preparation overrides"
    saved = {"steps": 20, "batch_size": 36, "micro_batches": 8, "data_workers": 1, "data_prefetch": 3,
             "seed": 19, "consistency": 1.0}
    inherited = cli.resume_config(saved, _parse(cli, ["--run", "runs/example"]))
    assert (inherited.data_workers, inherited.data_prefetch) == (1, 3)
    opts = _parse(cli, ["--run", "runs/example", "--data-workers", "0", "--data-prefetch", "1", "--micro-batches", "4"])
    changed = cli.resume_config(saved, opts)
    assert (changed.data_workers, changed.data_prefetch, changed.micro_batches) == (0, 1, 4)
    assert (changed.steps, changed.batch_size, changed.seed, changed.consistency) == (20, 36, 19, 1.0)
    assert changed.accumulate_gradients
    assert saved["data_workers"] == 1 and saved["micro_batches"] == 8


def test_old_resume_configs_get_defaults_and_invalid_overrides_are_refused():
    cli = _cli("resume")
    assert hasattr(cli, "resume_config"), "resume configuration does not support preparation overrides"
    old = {"steps": 20, "batch_size": 36, "micro_batches": 8}
    cfg = cli.resume_config(old, _parse(cli, ["--run", "runs/example"]))
    assert (cfg.data_workers, cfg.data_prefetch) == (2, 2)
    for flags in (["--data-workers", "-1"], ["--data-prefetch", "0"], ["--micro-batches", "0"]):
        try:
            cli.resume_config(old, _parse(cli, ["--run", "runs/example", *flags]))
        except ValueError:
            continue
        raise AssertionError("invalid resume execution limits accepted")


if __name__ == "__main__":
    run(globals())
