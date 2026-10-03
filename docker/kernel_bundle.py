"""Prepare one pinned GPU profile online; validate and expose it offline.

No GPU is required to prepare a bundle. The build downloads only the selected
Linux/PyTorch/CUDA variants and copies real files out of the temporary Hub cache.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import platform
import shutil
import tempfile
import tomllib

CONFIG = Path(__file__).with_name("kernels.toml")


def profile_config(name):
    config = tomllib.loads(CONFIG.read_text())
    if name not in config["profiles"]:
        raise ValueError(f"unknown kernel profile {name!r}; choose from {', '.join(config['profiles'])}")
    profile = dict(config["profiles"][name], name=name, environment=config["environment"])
    kernels = [dict(config["kernels"][key], directory=key) for key in profile["kernels"]]
    return profile, kernels


def check_environment(expected):
    import torch

    actual = {"torch": torch.__version__.split("+")[0], "cuda": torch.version.cuda,
              "python": ".".join(platform.python_version_tuple()[:2]),
              "system": platform.system(), "machine": platform.machine()}
    if actual != expected:
        raise ValueError(f"kernel bundle environment mismatch: expected {expected}, got {actual}")


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").digest()


def verify_kernel(directory, metadata_hash=None):
    metadata_path = directory / "metadata.json"
    actual = file_hash(metadata_path).hex()
    if metadata_hash is not None and metadata_hash != actual:
        raise ValueError(f"kernel metadata checksum mismatch: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    digest = metadata["digest"]
    if digest["algorithm"] != "sha256" or not digest["files"]:
        raise ValueError(f"kernel has no supported checksums: {directory}")
    root = directory.resolve()
    for name, expected in digest["files"].items():
        path = directory / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"kernel file must be self-contained: {path}")
        if base64.b64encode(file_hash(path)).decode() != expected:
            raise ValueError(f"kernel checksum mismatch: {path}")
    return actual


def prepare(name, destination):
    from huggingface_hub import snapshot_download

    profile, kernels = profile_config(name)
    check_environment(profile["environment"])
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"bundle destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"profile": name, "capabilities": profile["capabilities"], "attention": profile["attention"],
                "environment": profile["environment"], "kernels": {}}
    with tempfile.TemporaryDirectory(prefix=".kernel-build-", dir=destination.parent) as temporary:
        temporary = Path(temporary)
        bundle = temporary / "bundle"
        bundle.mkdir()
        for kernel in kernels:
            print(f"fetching {kernel['repo']}@{kernel['revision']} / {kernel['variant']}", flush=True)
            snapshot = snapshot_download(kernel["repo"], repo_type="kernel", revision=kernel["revision"],
                                         allow_patterns=[f"build/{kernel['variant']}/*"],
                                         ignore_patterns=["*.pyc", "**/__pycache__/**"],
                                         cache_dir=temporary / "hub", token=False)
            target = bundle / kernel["directory"]
            shutil.copytree(Path(snapshot) / "build" / kernel["variant"], target, symlinks=False)
            checksum = verify_kernel(target)
            manifest["kernels"][kernel["repo"]] = dict(kernel, metadata_sha256=checksum)
        (bundle / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
        bundle.rename(destination)
    print(f"prepared offline profile {name}: {destination}", flush=True)


def runtime_environment(directory, capability):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "bundle.json").read_text())
    check_environment(manifest["environment"])
    sm = capability[0] * 10 + capability[1]
    if sm not in manifest["capabilities"]:
        raise ValueError(f"profile {manifest['profile']} supports SM {manifest['capabilities']}, got SM{sm}")
    mappings = []
    for repo, kernel in manifest["kernels"].items():
        local = directory / kernel["directory"]
        if not local.resolve().is_relative_to(directory):
            raise ValueError(f"kernel directory is outside the bundle: {local}")
        verify_kernel(local, kernel["metadata_sha256"])
        mappings.append(f"{repo}={local}")
    return manifest, {"LOCAL_KERNELS": ":".join(mappings), "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--dest", type=Path, default=Path("/opt/kernels"))
    args = parser.parse_args()
    prepare(args.profile, args.dest)


if __name__ == "__main__":
    main()
