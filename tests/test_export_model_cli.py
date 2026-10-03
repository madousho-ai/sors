"""The export command produces a locally loadable complete model from a checkpoint."""

import os
from pathlib import Path
import subprocess
import tempfile

import torch

from _runner import run
from test_decision import batch, tiny_backbone, tiny_model
from sors.core.checkpoint import save_trained
from sors.training.loop import TrainConfig

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/export-model.py"


def test_export_command_restores_and_saves_one_complete_model():
    assert SCRIPT.is_file(), "complete-model export command is missing"
    from sors.serve.engine import load_engine

    model, tok, d_ids, ids = tiny_model("minimal", trainable="full")
    model.eval()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(torch.randn_like(parameter) * 0.01)
    data = batch(model, tok, d_ids)
    with torch.inference_mode():
        expected = model.forward_batch(data)
    with tempfile.TemporaryDirectory() as directory:
        base, checkpoint, output = (Path(directory) / name for name in ("base", "trained.safetensors", "release"))
        tiny_backbone(tok).save_pretrained(base)
        tok.save_pretrained(base)
        save_trained(model, ids, TrainConfig(loss="menu"), checkpoint)
        command = [os.sys.executable, str(SCRIPT), "--init", str(checkpoint), "--base-model", str(base),
                   "--out", str(output), "--device", "cpu", "--dtype", "float32", "--local-files-only"]
        proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                              env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "HF_HUB_OFFLINE": "1"}, timeout=60)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert len(list(output.rglob("*.safetensors"))) == 1
        assert not list(output.rglob("*.py"))
        engine = load_engine(output, device="cpu", local_files_only=True)
        with torch.inference_mode():
            torch.testing.assert_close(engine.lm.forward_batch(data), expected, atol=0, rtol=0)
        again = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                               env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "HF_HUB_OFFLINE": "1"}, timeout=60)
        assert again.returncode != 0 and "exist" in (again.stdout + again.stderr).lower()


if __name__ == "__main__":
    run(globals())
