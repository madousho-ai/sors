"""The container's launch contract preserves caller paths and enforces offline startup."""

import importlib.util
from pathlib import Path

from _runner import run


def load(name, path):
    assert path.is_file(), f"missing container entry point: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_container_flags_keep_model_overrides_and_force_offline_warmup():
    root = Path(__file__).resolve().parents[1]
    entry = load("container_entrypoint", root / "docker" / "entrypoint.py")
    serve = load("container_serve", root / "scripts" / "serve.py")
    args = serve.build_parser().parse_args(entry.server_arguments(
        ["--init", "/payload/run.safetensors", "--base-model", "/payload/base", "--model-name", "release",
         "--port", "8080", "--allow-kernel-download"], "flash_attention_3"))
    assert args.init == "/payload/run.safetensors" and args.base_model == "/payload/base"
    assert args.model_name == "release" and args.port == 8080 and args.host == "0.0.0.0"
    assert args.attn_implementation == "flash_attention_3"
    assert args.local_files_only and args.warmup and not args.allow_kernel_download


def test_container_defaults_to_one_complete_model_directory():
    root = Path(__file__).resolve().parents[1]
    entry = load("complete_container_entrypoint", root / "docker" / "entrypoint.py")
    serve = load("complete_container_serve", root / "scripts" / "serve.py")
    args = serve.build_parser().parse_args(entry.server_arguments([], "flash_attention_2"))
    assert args.init == "/models", "container still requires a separate training checkpoint"
    assert args.base_model is None, "container still ships a separate base model"
    assert args.local_files_only and args.warmup and not args.allow_kernel_download


if __name__ == "__main__":
    run(globals())
