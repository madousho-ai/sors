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
import subprocess
import tempfile
import time

import torch
from sors.data.paths import asset_path, datasets_root

FORMAT = "sors-training-state-v1"
LEGACY_FORMAT = "decidophobia-training-state-v1"


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
        if meta.get("format") not in (FORMAT, LEGACY_FORMAT):
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


SAMPLING_SOURCES = ("scripts/train.py", "src/sors/data/synth_v5.py", "src/sors/core/menu.py",
                    "src/sors/core/prompt.py", "src/sors/serve/menus.py",
                    "src/sors/data/paths.py", "src/sors/training/resume.py")
V5 = "synth-intents-v5.3"


def _git(repo, *args) -> bytes:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    if result.returncode:
        raise ValueError(f"cannot verify sampling revision in {repo}: {result.stderr.decode().strip()}")
    return result.stdout


def _revision(repo, revision) -> str:
    return _git(repo, "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}").decode().strip()


def sampling_fingerprint(repo, datasets_dir=None, *, code_commit=None, data_commit=None) -> str:
    """Hash logical paths and bytes across both checkouts, independent of their locations."""
    root, assets = pathlib.Path(repo), datasets_root(datasets_dir)
    code_ref = _revision(root, code_commit) if code_commit is not None else None
    data_ref = _revision(assets, data_commit) if data_commit is not None else None
    if data_ref:
        data_names = [p.decode() for p in _git(assets, "ls-tree", "-rz", "--name-only", data_ref, "--", V5).split(b"\0")
                      if p and pathlib.PurePosixPath(p.decode()).parent == pathlib.PurePosixPath(V5)
                      and p.endswith(b".json")]
    else:
        data_names = [f"{V5}/{p.name}" for p in asset_path(V5, assets).glob("*.json")]
    if not data_names:
        raise ValueError(f"{assets / V5}: no dataset JSON files; check --datasets-dir")
    h = hashlib.sha256()
    files = [(name, root, name, code_ref) for name in SAMPLING_SOURCES]
    files += [(f"datasets/{name}", assets, name, data_ref) for name in [f"{V5}/schema.py", *sorted(data_names)]]
    for logical, directory, name, revision in files:
        h.update(logical.encode())
        h.update(b"\0")
        h.update(_git(directory, "cat-file", "blob", f"{revision}:{name}") if revision else (directory / name).read_bytes())
    return h.hexdigest()


def _split_compat(repo) -> dict:
    path = pathlib.Path(repo) / "src/sors/training/dataset_split_compat.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def verify_sampling_fingerprint(repo, datasets_dir, expected: str) -> str:
    current = sampling_fingerprint(repo, datasets_dir)
    if current == expected:
        return current
    migration = _split_compat(repo)
    if current == migration.get("current_fingerprint") and expected in migration.get("legacy_fingerprints", []):
        return current
    raise ValueError("sampling code or dataset changed since this training checkpoint")


def verify_sampling_commits(repo, datasets_dir, data_commit: str, *, code_commit=None) -> None:
    current = sampling_fingerprint(repo, datasets_dir)
    if code_commit is not None:
        expected = sampling_fingerprint(repo, datasets_dir, code_commit=code_commit, data_commit=data_commit)
        if expected == current:
            return
        raise ValueError("sampling code or dataset differs from the specified commits")
    migration = _split_compat(repo)
    matches = [c for c in migration.get("legacy_commits", []) if c.startswith(data_commit)] if len(data_commit) >= 7 else []
    if len(matches) == 1 and current == migration.get("current_fingerprint"):
        return
    raise ValueError("legacy sampling revision cannot be verified; specify both --expected-code-commit and "
                     "--expected-data-commit for the split repositories")


def sampling_provenance(repo, datasets_dir=None) -> dict:
    out = {"datasets_dir": str(datasets_root(datasets_dir))}
    for label, directory in (("code", pathlib.Path(repo).resolve()), ("data", datasets_root(datasets_dir))):
        try:
            top = pathlib.Path(_git(directory, "rev-parse", "--show-toplevel").decode().strip()).resolve()
            if top != directory:
                raise ValueError("asset export is inside another repository")
            out[f"{label}_commit"] = _revision(directory, "HEAD")
            out[f"{label}_dirty"] = bool(_git(directory, "status", "--porcelain", "--untracked-files=all"))
        except (ValueError, FileNotFoundError):
            out[f"{label}_commit"], out[f"{label}_dirty"] = None, None
    return out
