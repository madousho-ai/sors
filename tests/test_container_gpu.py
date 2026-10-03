"""Offline container GPU smoke: tiny local checkpoints, real serving, no downloads.

Run inside a runtime image with /tests mounted and /app/docker on PYTHONPATH.
--payload-dir also exports test-only assets for checking the real HTTP entrypoint.
"""

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import tempfile

import torch


def exercise(root, attention):
    from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3_5TextConfig, Qwen3_5ForCausalLM
    from test_decision import tokenizer
    from sors.core.checkpoint import save_trained
    from sors.core.decision import DecisionConfig
    from sors.core.model import prepare_model
    from sors.serve.api import Choice, Noul
    from sors.serve.engine import load_engine
    from sors.training.loop import TrainConfig

    spec = importlib.util.spec_from_file_location("container_gpu_serve", "/app/scripts/serve.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    questions = {
        "color": Choice(type="choice", instructions="Which color?",
                        criteria={"red": None, "long blue": None, "long long green": None}),
        "red": Noul(type="noul", instructions="Does this context mention the color red?"),
    }
    for family in ("qwen3-slots", "qwen35-candidate"):
        torch.manual_seed(51)
        directory = root / family
        directory.mkdir(parents=True)
        tok, codes, ids = tokenizer()
        if family == "qwen3-slots":
            source = Qwen3ForCausalLM(Qwen3Config(
                vocab_size=len(tok) + 4, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                num_attention_heads=2, num_key_value_heads=1, head_dim=64, use_cache=False))
            decision, trainable = None, "full"
        else:
            source = Qwen3_5ForCausalLM(Qwen3_5TextConfig(
                vocab_size=len(tok) + 4, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                num_attention_heads=2, num_key_value_heads=1, head_dim=32,
                linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=32,
                linear_value_head_dim=32, layer_types=["linear_attention", "full_attention"],
                tie_word_embeddings=True, use_cache=False))
            decision, trainable = DecisionConfig(kind="candidate", blocks=0, dim=32, heads=4), "decision-only"
        base = directory / "base"
        source.save_pretrained(base)
        tok.save_pretrained(base)
        model = prepare_model(source, ids, None, None, 0.0, trainable=trainable, decision=decision)
        checkpoint = directory / "trained.safetensors"
        save_trained(model, ids, TrainConfig(layout="context-first"), checkpoint)
        del source, model

        options = dict(device="cuda", dtype=torch.bfloat16, allow_kernel_download=False,
                       local_files_only=True, max_tokens=2048, max_batch_tokens=4096)
        if family == "qwen35-candidate":
            options["candidate_prefix_cache"] = "on"
        reference = load_engine(checkpoint, base, attn_implementation="sdpa", **options)
        actual = load_engine(checkpoint, base, attn_implementation=attention, **options)
        cli.warmup(actual)
        for repeats in (1, 16, 96):
            state = "red blue context. " * repeats
            want, got = reference.evaluate(state, questions), actual.evaluate(state, questions)
            again = actual.evaluate(state, questions)
            for name in questions:
                expected, result = torch.tensor(want.probs[name]), torch.tensor(got.probs[name])
                assert torch.isfinite(result).all() and abs(float(result.sum()) - 1) < 1e-4
                torch.testing.assert_close(result, expected, rtol=0.03, atol=0.01)
                torch.testing.assert_close(result, torch.tensor(again.probs[name]), rtol=0, atol=0)
            print(f"PASS offline serving {family} state_repeats={repeats} backend={attention}", flush=True)
        del actual, reference
        gc.collect()
        torch.cuda.empty_cache()

    from kernels import get_loaded_kernels
    loaded_paths = [str(kernel.module.__file__) for kernel in get_loaded_kernels()]
    assert any(path.startswith("/opt/kernels/conv/") for path in loaded_paths), loaded_paths
    assert any(path.startswith("/opt/kernels/fa") for path in loaded_paths), loaded_paths
    print(json.dumps({"loaded_local_kernels": loaded_paths}), flush=True)


def main():
    from kernel_bundle import runtime_environment

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-dir", type=Path)
    args = parser.parse_args()
    manifest, environment = runtime_environment("/opt/kernels", torch.cuda.get_device_capability())
    os.environ.update(environment)
    if args.payload_dir:
        exercise(args.payload_dir, manifest["attention"])
    else:
        with tempfile.TemporaryDirectory(prefix="sors-container-gpu-") as temporary:
            exercise(Path(temporary), manifest["attention"])


if __name__ == "__main__":
    main()
