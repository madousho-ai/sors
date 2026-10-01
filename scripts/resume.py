#!/usr/bin/env python
"""在原 run 续训。保留原 train.py 与数据版本，恢复从 completed step 后的一次更新。

旧权重档须显式给 --allow-optimizer-reset 与 --expected-data-commit；原参数从 TensorBoard 读取。
之后默认从 checkpoints/latest.trainstate.safetensors 恢复完整状态。--stop-after N 可暂停并保存现场，
原目标总步数/LR 日程保持不变。所有验证和训练应在原服务器进行。

首次恢复旧档（先确认原数据/采样代码与该 commit 相同）：
  PYTHONPATH=src .venv/bin/python scripts/resume.py --run runs/<原run> \
    --checkpoint runs/<原run>/checkpoints/step-01500.safetensors \
    --allow-optimizer-reset --expected-data-commit c8a2330
后续完整恢复：
  PYTHONPATH=src .venv/bin/python scripts/resume.py --run runs/<原run>
latest.trainstate.safetensors 原子替换，仅保留最新完整状态；推理权重仍按原 save_every 单独存档。
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from dataclasses import asdict

import torch
from transformers import AutoTokenizer

from decidophobia.core.attention import add_attention_arguments, load_causal_lm
from decidophobia.core.checkpoint import load_trained, save_trained
from decidophobia.core.model import adapter_config, prepare_model
from decidophobia.core.tokens import install_context_tokens, install_d_tokens, install_type_tokens
from decidophobia.training.loop import TrainConfig, train
from decidophobia.training.resume import read_state, resume_writer, rollback_history, sampling_fingerprint, save_state
from decidophobia.training.thermal import ThermalGuard

ROOT = pathlib.Path(__file__).resolve().parents[1]


def original_args(out: pathlib.Path) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(out / "tb"), size_guidance={"tensors": 0})
    ea.Reload()
    events = ea.Tensors("args/text_summary")
    return json.loads(events[0].tensor_proto.string_val[0].decode())


def legacy_step(path: pathlib.Path, accepted: bool) -> int:
    if not accepted:
        raise ValueError("old weight checkpoints require explicit --allow-optimizer-reset")
    match = re.fullmatch(r"step-(\d+)\.safetensors", path.name)
    if not match:
        raise ValueError("old checkpoint must name its completed step: step-NNNNN.safetensors")
    return int(match.group(1))


def validate_resume_step(step: int, target: int, *, full: bool) -> None:
    if not 0 <= step <= target or (step == target and not full):
        raise ValueError("invalid resume step; completed runs need their full state for export recovery")


def _versions():
    import transformers
    return {"python": sys.version, "torch": str(torch.__version__), "transformers": transformers.__version__}


def _write_json(path, value):
    tmp = path.with_name(f".{path.name}.{os.getpid()}")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    add_attention_arguments(ap, resume=True)
    ap.add_argument("--run", type=pathlib.Path, required=True)
    ap.add_argument("--checkpoint", type=pathlib.Path)
    ap.add_argument("--allow-optimizer-reset", action="store_true")
    ap.add_argument("--expected-data-commit")
    ap.add_argument("--stop-after", type=int)
    ap.add_argument("--micro-batches", type=int)
    return ap


def main():
    opts = build_parser().parse_args()
    out = opts.run.resolve(strict=True)
    lock = (out / ".resume.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = (opts.checkpoint or out / "checkpoints/latest.trainstate.safetensors").resolve(strict=True)
    fingerprint = sampling_fingerprint(ROOT)
    full = path.name.endswith(".trainstate.safetensors")
    if full:
        state = read_state(path)
        meta = state["metadata"]
        if meta["sampling_fingerprint"] != fingerprint:
            raise ValueError("sampling code or dataset changed since this training checkpoint")
        if meta["versions"] != _versions():
            raise ValueError("Python/PyTorch/Transformers versions differ from the saved training environment")
        args = argparse.Namespace(**meta["args"])
        config = state["config"]
        revision = meta.get("model_revision")
    else:
        from safetensors import safe_open
        step = legacy_step(path, opts.allow_optimizer_reset)
        if not opts.expected_data_commit:
            raise ValueError("old checkpoints require --expected-data-commit to verify original sampler/data")
        sources = ["scripts/train.py", "src/decidophobia/data/synth_v5.py", "src/decidophobia/core/menu.py",
                   "src/decidophobia/core/prompt.py", "src/decidophobia/serve/menus.py", "datasets/synth-intents-v5.1"]
        subprocess.run(["git", "diff", "--exit-code", opts.expected_data_commit, "--", *sources],
                       cwd=ROOT, check=True)
        if path.parent != out / "checkpoints":
            raise ValueError("legacy checkpoint must belong to the original run's checkpoints directory")
        args = argparse.Namespace(**original_args(out))
        with safe_open(str(path), framework="pt", device="cpu") as f:
            config = json.loads(f.metadata()["config"])
        state = {"step": step, "config": config}
        revision = None
    if args.dataset not in ("synth-v5", "synth-v5.1"):
        raise ValueError("this recovery entry point currently verifies synth-v5.1 runs (legacy alias: synth-v5) only")
    cfg = TrainConfig(**config)
    cfg.accumulate_gradients = True
    if opts.micro_batches is not None:
        if opts.micro_batches < 1:
            raise ValueError("micro_batches must be positive")
        cfg.micro_batches = opts.micro_batches
    start = state["step"]
    validate_resume_step(start, cfg.steps, full=full)
    args.out = str(out)
    args.micro_batches = cfg.micro_batches
    spec = importlib.util.spec_from_file_location("original_train_cli", ROOT / "scripts/train.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    sample_fn, eval_sets, split = cli.build_data(args)
    tok = AutoTokenizer.from_pretrained(args.model, revision=revision, local_files_only=True)
    d_ids = install_d_tokens(tok)
    train_ids = d_ids + install_type_tokens(tok) + install_context_tokens(tok)
    if full and train_ids != meta["train_ids"]:
        raise ValueError("trainable token ids differ from the saved state")
    previous_attention = getattr(args, "attention", {})
    args.attn_implementation = opts.attn_implementation or previous_attention.get("backend", "sdpa")
    args.allow_kernel_download = (getattr(args, "allow_kernel_download", False)
                                  if opts.allow_kernel_download is None else opts.allow_kernel_download)
    lm, attention = load_causal_lm(args.model, revision=revision, local_files_only=True,
                                   attn_implementation=args.attn_implementation,
                                   allow_kernel_download=args.allow_kernel_download)
    args.attention = asdict(attention)
    model_revision = getattr(lm.config, "_commit_hash", None)
    m = prepare_model(lm, train_ids, args.lora_r, args.lora_alpha, args.lora_dropout,
                      trainable=args.trainable, grad_ckpt=args.grad_ckpt)
    if full:
        if adapter_config(m) != meta["adapter"]:
            raise ValueError("adapter configuration differs from the saved state")
    else:
        load_trained(m, train_ids, path)
    history = rollback_history(out, start)
    if not full:
        if not history or history[-1]["step"] != start:
            raise ValueError("legacy recovery needs the evaluation record for its checkpoint step")
        state["history"] = history
    metadata = {"args": vars(args), "sampling_fingerprint": fingerprint, "versions": _versions(),
                "train_ids": train_ids, "adapter": adapter_config(m), "model_revision": model_revision}
    record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "checkpoint": str(path), "completed_step": start,
              "target_step": cfg.steps, "optimizer_reset": not full, "accumulate_gradients": True,
              "micro_batches": cfg.micro_batches, "sampling_fingerprint": fingerprint,
              "previous_attention": previous_attention, "attention": args.attention}
    with (out / "resume.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    print("resume: " + json.dumps(record), flush=True)
    print(f"resume: next update={start + 1}, target={cfg.steps}, batch_size={cfg.batch_size}", flush=True)
    latest = out / "checkpoints/latest.trainstate.safetensors"
    completed = start
    final_info = dict(state.get("resume_info", {}))

    def weights(step):
        dest = out / "checkpoints" / f"step-{step:05d}.safetensors"
        tmp = dest.with_suffix(".writing")
        save_trained(m, train_ids, cfg, tmp)
        os.replace(tmp, dest)
        print(f"step {step:5d} saved weights {dest}", flush=True)

    def training_state(value):
        nonlocal completed, final_info
        completed = value["step"]
        final_info = dict(value["resume_info"])
        print(f"step {completed:5d} saving complete training state", flush=True)
        save_state({**value, "metadata": metadata}, latest)
        print(f"step {completed:5d} saved complete training state {latest}", flush=True)

    with resume_writer(out, start) as writer:
        writer.add_text("resume", json.dumps(record, indent=2), start)
        history = train(m, tok, d_ids, sample_fn, eval_sets, cfg, log_path=out / "log.jsonl", writer=writer,
                        guard=ThermalGuard(max_c=args.temp_max, cooldown_s=args.temp_cooldown),
                        on_checkpoint=weights, resume=state, on_state=training_state, stop_after=opts.stop_after)
    if completed == cfg.steps:
        tmp = out / "trained.writing"
        save_trained(m, train_ids, cfg, tmp)
        os.replace(tmp, out / "trained.safetensors")
        _write_json(out / "result.json", {"args": vars(args), "train_config": asdict(cfg), "split": split,
                    "history": history, "resume_info": final_info,
                    "trainable_params": sum(p.numel() for p in m.parameters() if p.requires_grad),
                    "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)})
    print(f"resume: paused/completed at {completed}/{cfg.steps}; peak {torch.cuda.max_memory_allocated()/2**30:.3f} GiB",
          flush=True)


if __name__ == "__main__":
    main()
