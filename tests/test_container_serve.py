"""Serve a local base + checkpoint with downloads disabled and a real CPU model."""

import importlib.util
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from _runner import run
from test_decision import tiny_model, tiny_backbone
from sors.core.checkpoint import save_trained
from sors.serve.api import Choice
from sors.serve.engine import Engine, load_engine
from sors.training.loop import TrainConfig


def test_local_only_engine_restores_checkpoint_and_answers_without_network():
    model, tok, codes, ids = tiny_model("minimal")
    question = {"q": Choice(type="choice", instructions="Which color?", criteria={"red": None, "blue": None})}
    expected = Engine(model, tok, codes).evaluate("red context", question)
    with tempfile.TemporaryDirectory() as directory:
        base, checkpoint = Path(directory) / "base", Path(directory) / "trained.safetensors"
        tiny_backbone(tok).save_pretrained(base)
        tok.save_pretrained(base)
        save_trained(model, ids, TrainConfig(layout="context-first"), checkpoint)
        with patch("httpx.Client.send", side_effect=AssertionError("offline service attempted HTTP")):
            try:
                engine = load_engine(checkpoint, base, device="cpu", dtype=torch.float32,
                                     attn_implementation="sdpa", allow_kernel_download=False,
                                     local_files_only=True)
            except TypeError as exc:
                raise AssertionError(f"load_engine does not expose offline loading: {exc}") from exc
            result = engine.evaluate("red context", question)
        torch.testing.assert_close(torch.tensor(result.probs["q"]), torch.tensor(expected.probs["q"]))


def test_warmup_rejects_nonfinite_real_model_output():
    path = Path(__file__).resolve().parents[1] / "scripts" / "serve.py"
    spec = importlib.util.spec_from_file_location("container_serve_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert hasattr(cli, "warmup"), "serve warmup is missing"
    model, tok, codes, _ = tiny_model("minimal")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(float("nan"))
    try:
        cli.warmup(Engine(model, tok, codes))
    except ValueError as exc:
        assert "probab" in str(exc).lower()
    else:
        raise AssertionError("warmup accepted nonfinite predictions")


def test_complete_model_payload_restores_and_answers_with_downloads_blocked():
    from sors.core.pretrained import save_pretrained

    for architecture in ("minimal", "candidate"):
        model, tok, codes, _ = tiny_model(architecture, trainable="full")
        question = {"q": Choice(type="choice", instructions="Which color?", criteria={"red": None, "blue": None})}
        expected = Engine(model, tok, codes).evaluate("red context", question)
        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory) / "payload"
            save_pretrained(model, tok, TrainConfig(layout="context-first"), payload)
            assert [path.name for path in payload.glob("*.safetensors")] == ["model.safetensors"]
            with patch("httpx.Client.send", side_effect=AssertionError("offline service attempted HTTP")):
                engine = load_engine(payload, device="cpu", attn_implementation="sdpa",
                                     allow_kernel_download=False, local_files_only=True)
                result = engine.evaluate("red context", question)
            assert result.probs == expected.probs


if __name__ == "__main__":
    run(globals())
