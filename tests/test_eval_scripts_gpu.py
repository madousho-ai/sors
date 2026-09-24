"""scripts/eval-massive.py 与 scripts/eval-invariance.py 端到端: 喂一份 r4 alpha8 的 trained.pt,
脚本得照档里记的形状搭 LoRA 空壳, 不靠命令行复述. 每个脚本只评几道题. 要 0.6B 和 GPU.

跑:  OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python tests/test_eval_scripts_gpu.py
"""

import json
import os
import pathlib
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _runner import run  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL = "Qwen/Qwen3-0.6B-Base"
_TMP = pathlib.Path(tempfile.mkdtemp(prefix="eval-scripts-"))
_state: dict = {}


def _checkpoint() -> str:
    """0.6B 在 CPU 上走 prepare_model (r4 alpha8, 离开默认的 r8 alpha16) 与 save_trained, 存一份 trained.pt."""
    if not _state:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from decidophobia.model import prepare_model
        from decidophobia.tokens import install_d_tokens, install_type_tokens
        from decidophobia.train import TrainConfig, save_trained

        tok = AutoTokenizer.from_pretrained(MODEL)
        ids = install_d_tokens(tok) + install_type_tokens(tok)
        lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
        m = prepare_model(lm, ids, lora_r=4, lora_alpha=8, lora_dropout=0.0)
        path = _TMP / "r4" / "trained.pt"
        path.parent.mkdir()
        save_trained(m, ids, TrainConfig(), path)
        _state["path"] = str(path)
    return _state["path"]


def _script(name: str, *argv: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "HF_HUB_OFFLINE": "1"}
    return subprocess.run([sys.executable, str(ROOT / "scripts" / name), *argv], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=600)


def test_eval_massive_evaluates_a_checkpoint_trained_with_rank_4():
    """未训练基线 + 那份 r4 档, 各一条记录; 档的那条带着它的训练 config."""
    out = _TMP / "massive.json"
    p = _script("eval-massive.py", "--init", _checkpoint(), "--k", "10", "--limit", "8", "--out", str(out))
    assert p.returncode == 0, p.stderr[-2000:]
    recs = json.loads(out.read_text())["records"]
    assert [(r["tag"], r["n"]) for r in recs] == [("untrained", 8), ("r4", 8)], [(r["tag"], r["n"]) for r in recs]
    assert recs[1]["train_config"] is not None


def test_eval_invariance_evaluates_a_checkpoint_trained_with_rank_4():
    out = _TMP / "invariance.json"
    p = _script("eval-invariance.py", "--init", _checkpoint(), "--limit", "2", "--batch-size", "1", "--out", str(out))
    assert p.returncode == 0, p.stderr[-2000:]
    got = json.loads(out.read_text())
    assert got["records"][0]["variant"] == "base" and len(got["gold"]) == 2


if __name__ == "__main__":
    run(globals())
