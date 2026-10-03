"""Validate the offline GPU bundle, then exec the normal serving CLI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def server_arguments(arguments, attention):
    return ["--init", "/models",
            "--model-name", "sors", "--host", "0.0.0.0", "--attn-implementation", attention,
            *arguments, "--no-allow-kernel-download", "--local-files-only", "--warmup"]


def main():
    script = str(Path(__file__).resolve().parents[1] / "scripts" / "serve.py")
    if "--help" in sys.argv[1:] or "-h" in sys.argv[1:]:
        os.execv(sys.executable, [sys.executable, script, "--help"])

    import torch
    from kernel_bundle import runtime_environment

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", default="cuda")
    options, _ = parser.parse_known_args()
    device = torch.device(options.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this GPU image requires a visible NVIDIA GPU and a compatible host driver")
    manifest, environment = runtime_environment("/opt/kernels", torch.cuda.get_device_capability(device))
    os.environ.update(environment)
    print(json.dumps({"profile": manifest["profile"], "gpu": torch.cuda.get_device_name(device),
                      "capability": torch.cuda.get_device_capability(device),
                      "environment": manifest["environment"], "kernels": manifest["kernels"]}), flush=True)
    os.execv(sys.executable, [sys.executable, script, *server_arguments(sys.argv[1:], manifest["attention"])])


if __name__ == "__main__":
    main()
