"""训练现场的 safetensors 存档、历史回退与 TensorBoard 续写。

完整状态独立于推理权重档：一个原子文件内含模型、master、optimizer、scheduler、RNG 和统计窗口。
抽样器使用原数据/代码/seed 与完成的调用数重建；恢复后校验采样 RNG 的完整状态。
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import tempfile
import time

import torch

FORMAT = "decidophobia-training-state-v1"


def save_state(state: dict, path) -> None:
    """独立 CPU 快照 + safetensors + JSON 结构，保留 tuple 与整数 dict key；原子替换。"""
    from safetensors.torch import save_file

    tensors = {}

    def pack(value):
        if isinstance(value, torch.Tensor):
            name = f"tensor_{len(tensors):06d}"
            tensors[name] = value.detach().to(device="cpu", copy=True).contiguous()
            return {"type": "tensor", "value": name}
        if isinstance(value, dict):
            return {"type": "dict", "value": [[pack(k), pack(v)] for k, v in value.items()]}
        if isinstance(value, (tuple, list)):
            return {"type": "tuple" if isinstance(value, tuple) else "list", "value": [pack(v) for v in value]}
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(f"unsupported training-state value: {type(value)}")

    structure = json.dumps(pack(state), separators=(",", ":"))
    path = pathlib.Path(path)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        save_file(tensors, tmp, metadata={"format": FORMAT, "structure": structure})
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_state(path) -> dict:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as f:
        meta = f.metadata()
        if meta.get("format") != FORMAT:
            raise ValueError("this is not a complete training-state checkpoint")

        def unpack(node):
            if not isinstance(node, dict):
                return node
            kind, value = node["type"], node["value"]
            if kind == "tensor":
                return f.get_tensor(value)
            if kind == "dict":
                return {unpack(k): unpack(v) for k, v in value}
            if kind in ("tuple", "list"):
                xs = [unpack(v) for v in value]
                return tuple(xs) if kind == "tuple" else xs
            raise ValueError(f"unknown state node {kind!r}")

        return unpack(json.loads(meta["structure"]))


def rollback_history(out, completed_step: int) -> list[dict]:
    """先归档有失效记录的 JSONL，再保留已存档步数以内的评估记录。"""
    out = pathlib.Path(out)
    path = out / "log.jsonl"
    if not path.exists():
        return []
    text = path.read_text()
    lines = text.splitlines()
    records, partial = [], False
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1 or text.endswith("\n"):
                raise
            partial = True  # 中断追加留下的最后半行：先归档，后排除
    keep = [r for r in records if r["step"] <= completed_step]
    if partial or len(keep) != len(records):
        shutil.copy2(path, out / f"log.jsonl.before-resume-{time.time_ns()}")
        fd, tmp = tempfile.mkstemp(prefix=".log-resume-", dir=out)
        try:
            with os.fdopen(fd, "w") as f:
                for record in keep:
                    f.write(json.dumps(record) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    return keep


def resume_writer(out, completed_step: int):
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(log_dir=str(pathlib.Path(out) / "tb"), purge_step=completed_step + 1)


def sampling_fingerprint(repo) -> str:
    """本次 v5 采样所需的数据与代码；保护恢复时的序列和随机调用语义。"""
    root = pathlib.Path(repo)
    names = ["scripts/train.py", "src/decidophobia/data/synth_v5.py", "src/decidophobia/core/menu.py",
             "src/decidophobia/core/prompt.py", "src/decidophobia/serve/menus.py",
             "datasets/synth-intents-v5.1/schema.py"]
    files = [root / n for n in names] + sorted((root / "datasets/synth-intents-v5.1").glob("*.json"))
    h = hashlib.sha256()
    for path in files:
        h.update(str(path.relative_to(root)).encode())
        h.update(b"\0")
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()
