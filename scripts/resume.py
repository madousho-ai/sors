#!/usr/bin/env python
"""在原 run 续训。保留原 train.py 与数据版本，恢复从 completed step 后的一次更新。

旧权重档须显式给 --allow-optimizer-reset 与 --expected-data-commit；拆仓后的版本同时给 --expected-code-commit。
--datasets-dir 选择数据仓库；原参数从 TensorBoard 读取。迁移兼容范围见 DATASETS.md。
之后默认从 checkpoints/latest.trainstate.safetensors 恢复完整状态。--stop-after N 可暂停并保存现场，
原目标总步数/LR 日程保持不变。所有验证和训练应在原服务器进行。

首次恢复旧档（先确认原数据/采样代码与该 commit 相同）：
  PYTHONPATH=src .venv/bin/python scripts/resume.py --run runs/<原run> \
    --checkpoint runs/<原run>/checkpoints/step-01500.safetensors \
    --allow-optimizer-reset --expected-code-commit CODE_REV --expected-data-commit DATA_REV
后续完整恢复：
  PYTHONPATH=src .venv/bin/python scripts/resume.py --run runs/<原run>
latest.trainstate.safetensors 原子替换，仅保留最新完整状态；推理权重仍按原 save_every 单独存档。
--data-workers / --data-prefetch / --data-backend / --device-prefetch 可覆盖后台准备方式与预取上限；
默认继承存档，旧档默认 2/2、子进程、显存预取 1。
预取批次按原抽样顺序消费；完整状态里的采样 RNG 对应已完成训练的批次。
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
import pathlib
import re
import sys
import time
from dataclasses import asdict

import torch
from transformers import AutoTokenizer

from sors.core.attention import add_attention_arguments, load_causal_lm
from sors.core.checkpoint import checkpoint_architecture, load_trained, save_trained
from sors.core.decision import architecture_config, decision_config
from sors.core.model import adapter_config, prepare_model
from sors.core.tokens import install_context_tokens, install_d_tokens, install_type_tokens
from sors.data.paths import ENV, LEGACY_ENV, add_datasets_argument, datasets_root
from sors.training.loop import TrainConfig, train
from sors.training.prefetch import validate_data_preparation
from sors.training.resume import (read_state, resume_writer, rollback_history, sampling_fingerprint, save_state,
                                         sampling_provenance, verify_sampling_commits, verify_sampling_fingerprint)
from sors.training.thermal import ThermalGuard

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
    add_attention_arguments(ap)
    add_datasets_argument(ap)
    ap.add_argument("--run", type=pathlib.Path, required=True)
    ap.add_argument("--checkpoint", type=pathlib.Path)
    ap.add_argument("--allow-optimizer-reset", action="store_true")
    ap.add_argument("--expected-data-commit")
    ap.add_argument("--expected-code-commit", help="code revision paired with --expected-data-commit for split repositories")
    ap.add_argument("--stop-after", type=int)
    ap.add_argument("--micro-batches", type=int)
    ap.add_argument("--micro-tokens", type=int, help="覆盖每组补齐后的 token 预算 (>= 0)；0 = 按 --micro-batches 组数")
    ap.add_argument("--data-workers", type=int, help="覆盖后台 CPU 分词/组批线程数；0 = 同步准备")
    ap.add_argument("--data-prefetch", type=int, help="覆盖提前准备的批数上限 (>= 1)")
    ap.add_argument("--data-backend", choices=["process", "thread"], help="覆盖后台准备方式：子进程或线程")
    ap.add_argument("--device-prefetch", type=int, help="覆盖 CUDA 上提前拷进显存的批数 (>= 0)")
    return ap


def resume_config(config, opts):
    """Restore training semantics while allowing execution-only overrides."""
    cfg = TrainConfig(**config)
    cfg.accumulate_gradients = True
    for key in ("micro_batches", "micro_tokens", "data_workers", "data_prefetch", "data_backend", "device_prefetch"):
        value = getattr(opts, key)
        if value is not None:
            setattr(cfg, key, value)
    if cfg.micro_batches < 1:
        raise ValueError("micro_batches must be positive")
    if cfg.micro_tokens < 0:
        raise ValueError("micro_tokens must be nonnegative")
    validate_data_preparation(cfg.data_workers, cfg.data_prefetch, cfg.data_backend)
    if cfg.device_prefetch < 0:
        raise ValueError("device_prefetch must be nonnegative")
    return cfg


def prepare_resume_model(lm, train_ids, args, state, path, *, full):
    """Build the recorded computation graph before restoring its optimizer state."""
    if full:
        meta = state["metadata"]
        architecture = meta.get("architecture", state.get("architecture", {"kind": "slots"}))
        if state.get("architecture", {"kind": "slots"}) != architecture:
            raise ValueError("training state has conflicting architecture metadata")
    else:
        architecture = checkpoint_architecture(path)
        state["architecture"] = architecture
    m = prepare_model(lm, train_ids, args.lora_r, args.lora_alpha, args.lora_dropout,
                      trainable=args.trainable, grad_ckpt=args.grad_ckpt, decision=decision_config(architecture))
    if full:
        if adapter_config(m) != meta["adapter"]:
            raise ValueError("adapter configuration differs from the saved state")
    else:
        load_trained(m, train_ids, path)
    if architecture["kind"] == "candidate":
        m.set_candidate_prefix_cache(state.get("config", {}).get("candidate_prefix_cache", "off"))
    return m


def main():
    opts = build_parser().parse_args()
    out = opts.run.resolve(strict=True)
    lock = (out / ".resume.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = (opts.checkpoint or out / "checkpoints/latest.trainstate.safetensors").resolve(strict=True)
    full = path.name.endswith(".trainstate.safetensors")
    if full:
        state = read_state(path)
        meta = state["metadata"]
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
        if path.parent != out / "checkpoints":
            raise ValueError("legacy checkpoint must belong to the original run's checkpoints directory")
        args = argparse.Namespace(**original_args(out))
        with safe_open(str(path), framework="pt", device="cpu") as f:
            config = json.loads(f.metadata()["config"])
        state = {"step": step, "config": config}
        revision = None
    selected = opts.datasets_dir if opts.datasets_dir is not None else os.environ.get(
        ENV, os.environ.get(LEGACY_ENV, getattr(args, "datasets_dir", None)))
    args.datasets_dir = str(datasets_root(selected))
    if full:
        fingerprint = verify_sampling_fingerprint(ROOT, args.datasets_dir, meta["sampling_fingerprint"])
    else:
        verify_sampling_commits(ROOT, args.datasets_dir, opts.expected_data_commit, code_commit=opts.expected_code_commit)
        fingerprint = sampling_fingerprint(ROOT, args.datasets_dir)
    if args.dataset not in ("synth-v5", "synth-v5.3"):
        raise ValueError("this recovery entry point currently verifies synth-v5.3 runs (legacy alias: synth-v5) only")
    cfg = resume_config(config, opts)
    start = state["step"]
    validate_resume_step(start, cfg.steps, full=full)
    args.out = str(out)
    args.micro_batches, args.micro_tokens = cfg.micro_batches, cfg.micro_tokens
    args.data_workers, args.data_prefetch = cfg.data_workers, cfg.data_prefetch
    args.data_backend, args.device_prefetch = cfg.data_backend, cfg.device_prefetch
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
    args.attn_implementation = opts.attn_implementation
    args.allow_kernel_download = opts.allow_kernel_download
    lm, attention = load_causal_lm(args.model, revision=revision, local_files_only=True,
                                   attn_implementation=args.attn_implementation,
                                   allow_kernel_download=args.allow_kernel_download)
    args.attention = asdict(attention)
    model_revision = getattr(lm.config, "_commit_hash", None)
    m = prepare_resume_model(lm, train_ids, args, state, path, full=full)
    history = rollback_history(out, start)
    if not full:
        if not history or history[-1]["step"] != start:
            raise ValueError("legacy recovery needs the evaluation record for its checkpoint step")
        state["history"] = history
    metadata = {"args": vars(args), "sampling_fingerprint": fingerprint, "versions": _versions(),
                  "sampling_provenance": sampling_provenance(ROOT, args.datasets_dir),
                 "train_ids": train_ids, "adapter": adapter_config(m), "architecture": architecture_config(m),
                 "model_revision": model_revision}
    record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "checkpoint": str(path), "completed_step": start,
              "target_step": cfg.steps, "optimizer_reset": not full, "accumulate_gradients": True,
              "micro_batches": cfg.micro_batches, "micro_tokens": cfg.micro_tokens, "sampling_fingerprint": fingerprint,
              "data_workers": cfg.data_workers, "data_prefetch": cfg.data_prefetch,
              "data_backend": cfg.data_backend, "device_prefetch": cfg.device_prefetch,
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
