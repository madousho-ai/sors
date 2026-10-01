"""Training attention selection. Optional CUDA kernels are loaded before model weights.

Native FlashAttention packages are optional. Training CLIs allow Hub downloads by
default; programmatic callers pass the permission explicitly. Auto falls back to
SDPA on dependency failures, while explicit requests fail early.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass, replace
from importlib.metadata import version

import torch
from packaging.version import Version

ATTENTION_CHOICES = ("auto", "sdpa", "eager", "flash_attention_2", "flash_attention_3", "flash_attention_4")
_HUB_KERNELS = {
    "flash_attention_2": "kernels-community/flash-attn2",
    "flash_attention_3": "kernels-community/vllm-flash-attn3",
    "flash_attention_4": "kernels-community/flash-attn4",
}


@dataclass(frozen=True)
class AttentionChoice:
    requested: str
    backend: str
    implementation: str
    device: str
    gpu: str | None
    capability: tuple[int, int] | None
    reason: str
    linear_attention: dict[str, str] | None = None


def _probe_backend(backend: str, allow_kernel_download: bool, *, capability=None) -> str:
    from transformers import utils
    # The cached lazy importer marks a backend loaded before imports succeed. Probe
    # the uncached importer so a failed attempt cannot poison later model loads.
    from transformers.modeling_flash_attention_utils import _lazy_imports

    available = getattr(utils, f"is_flash_attn_{backend[-1]}_available")
    if available():
        if backend == "flash_attention_4" and Version(version("flash-attn-4")) < Version("4.0.0b33"):
            raise ImportError("FlashAttention 4 requires flash-attn-4>=4.0.0b33 for the verified training paths")
        if backend == "flash_attention_2" and capability in ((10, 0), (12, 0)):
            if Version(version("flash-attn")) < Version("2.8.3.post1"):
                raise ImportError("Blackwell requires flash-attn>=2.8.3.post1 with the matching GPU build")
        # Import the compiled extension as well as checking distribution metadata.
        _lazy_imports(backend)
        return backend
    if allow_kernel_download and utils.is_kernels_available():
        if backend == "flash_attention_4" and capability == (12, 0):
            raise ImportError("SM120 needs native flash-attn-4>=4.0.0b33; the Hub FA4 v0 kernel supports SM9x/10x/11x")
        implementation = _HUB_KERNELS[backend]
        _lazy_imports(implementation)
        return implementation
    raise ImportError(
        f"{backend} native package unavailable; install a compatible native package, "
        "or install a Transformers-compatible kernels package and opt in with --allow-kernel-download"
    )


def resolve_attention(
    requested: str = "auto", *, device="cuda", dtype=torch.bfloat16,
    allow_kernel_download: bool = False, head_dim: int | None = None, attention_dropout: float = 0.0,
) -> AttentionChoice:
    if requested not in ATTENTION_CHOICES:
        raise ValueError(f"unknown attention backend {requested!r}; choose from {ATTENTION_CHOICES}")
    device = torch.device(device)
    capability, gpu = None, None
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"CUDA device {device} is unavailable")
        capability = torch.cuda.get_device_capability(device)
        gpu = torch.cuda.get_device_name(device)

    def result(backend, implementation, reason):
        return AttentionChoice(requested, backend, implementation, str(device), gpu, capability, reason)

    if requested in ("sdpa", "eager"):
        return result(requested, requested, "explicit backend")
    candidates = ()
    if torch.version.hip is None and dtype in (torch.bfloat16, torch.float16):
        if capability in ((8, 0), (8, 6), (8, 7), (8, 9)):
            candidates = ("flash_attention_2",)
        elif capability == (9, 0):
            candidates = ("flash_attention_3", "flash_attention_2")
        elif capability in ((10, 0), (12, 0)):
            candidates = ("flash_attention_4", "flash_attention_2")
    if head_dim is not None and (head_dim <= 0 or head_dim > 256 or head_dim % 8):
        candidates = ()
    if attention_dropout:
        candidates = tuple(c for c in candidates if c == "flash_attention_2")
        # FA2 backward with head_dim 256 + dropout is restricted on consumer GPUs.
        if head_dim is not None and head_dim > 192 and capability not in ((8, 0), (9, 0)):
            candidates = ()
    if requested != "auto":
        if requested not in candidates:
            raise ValueError(f"{requested} is unsupported for device={device}, capability={capability}, "
                             f"dtype={dtype}, head_dim={head_dim}, attention_dropout={attention_dropout}")
        candidates = (requested,)
    failures = []
    for backend in candidates:
        try:
            implementation = _probe_backend(backend, allow_kernel_download, capability=capability)
        except (ImportError, OSError, ValueError) as exc:
            failures.append(f"{backend}: {exc}")
            if requested != "auto":
                raise ValueError(f"requested {backend} is unavailable: {exc}") from exc
        else:
            return result(backend, implementation, "; ".join([*failures, "compatible kernel loaded"]))
    reason = "; ".join(failures) or (f"SDPA for device={device}, capability={capability}, dtype={dtype}, "
                                     f"head_dim={head_dim}, attention_dropout={attention_dropout}")
    return result("sdpa", "sdpa", reason)


def add_attention_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--attn-implementation", choices=ATTENTION_CHOICES, default="auto",
                        help="attention 后端；默认 auto 按 GPU 架构与可用依赖选择，缺少兼容内核时使用 SDPA。")
    parser.add_argument("--allow-kernel-download", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="默认允许 Transformers 从 kernels-community 下载预编译 attention 内核；"
                             "可用 --no-allow-kernel-download 关闭，需自行安装兼容 kernels")


def _configure_linear_attention(lm, *, allow_kernel_download: bool) -> dict[str, str] | None:
    """Qwen3.5 needs convolution kernels separately from full attention's FA2.

    Configure only those functions on this model. Transformers' default mapping
    points at mamba-ssm, whose CUDA wheel does not export these convolution APIs.
    Installed FLA continues to supply the Gated DeltaNet implementation.
    """
    if lm.config.get_text_config().model_type != "qwen3_5_text" or lm.device.type != "cuda":
        return None
    from transformers import KernelConfig, utils

    native_conv = utils.is_causal_conv1d_available()
    selected = {
        "causal_conv1d": "causal-conv1d" if native_conv else "torch",
        "gated_delta_rule": "flash-linear-attention" if utils.is_flash_linear_attention_available() else "torch",
    }
    if native_conv or not allow_kernel_download or not utils.is_kernels_available():
        return selected

    from kernels import Mode, get_kernel

    repo = "kernels-community/causal-conv1d"
    names = ("causal_conv1d_fn", "causal_conv1d_update")
    try:
        kernel = get_kernel(repo, version=2)
        if not all(callable(getattr(kernel, name, None)) for name in names):
            raise ImportError(f"{repo} does not provide both causal convolution functions")
    except (ImportError, OSError, ValueError) as exc:
        selected["fallback_reason"] = str(exc)
        return selected

    config = KernelConfig(
        kernel_mapping={name: {"cuda": (f"{repo}:{name}", {"version": 2})} for name in names},
        inherit_mapping=False,
    )
    config.sanitize_kernel_mapping(lm)
    config.create_compatible_mapping(lm)
    # KernelConfig builds a mapping for the current mode. Add both modes before
    # activating it, so train/eval and checkpoint replay use the same kernel.
    for devices in config.kernel_mapping.values():
        for modes in devices.values():
            implementation = next(iter(modes.values()))
            modes[Mode.TRAINING] = modes[Mode.INFERENCE] = implementation
    # Empty device maps explicitly retain every other layer's implementation,
    # including the installed FLA package, without inheriting unrelated kernels.
    for name in config.registered_layer_names.values():
        config.kernel_mapping.setdefault(name, {})
    lm.kernel_config = config
    lm.set_use_kernels(True)
    selected["causal_conv1d"] = f"{repo}@v2"
    return selected


def load_causal_lm(
    model_id, *, attn_implementation="auto", allow_kernel_download=False, device="cuda", dtype=torch.bfloat16,
    revision=None, local_files_only=False,
):
    """Load on a single target device and return the model plus its actual attention selection.

    Only optional-kernel probing may fall back. Model loading and CUDA runtime errors
    propagate, including OOM. No package installation is performed here.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    device = torch.device(device)
    config = AutoConfig.from_pretrained(model_id, revision=revision, local_files_only=local_files_only)
    text_config = config.get_text_config()
    head_dim = getattr(text_config, "head_dim", None)
    if head_dim is None and hasattr(text_config, "hidden_size") and hasattr(text_config, "num_attention_heads"):
        head_dim = text_config.hidden_size // text_config.num_attention_heads
    # HF availability checks inspect the current CUDA device. Match it to the load target.
    with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
        choice = resolve_attention(attn_implementation, device=device, dtype=dtype,
                                   allow_kernel_download=allow_kernel_download, head_dim=head_dim,
                                   attention_dropout=getattr(text_config, "attention_dropout", 0.0))
        lm = AutoModelForCausalLM.from_pretrained(
            model_id, config=config, revision=revision, local_files_only=local_files_only,
            dtype=dtype, attn_implementation=choice.implementation,
        ).to(device)
        choice = replace(choice, linear_attention=_configure_linear_attention(
            lm, allow_kernel_download=allow_kernel_download,
        ))
    actual = lm.config.get_text_config()._attn_implementation
    if actual != choice.implementation:
        raise ValueError(f"model changed requested attention {choice.implementation!r} to {actual!r}")
    print(f"attention: requested={choice.requested} actual={actual} device={device} "
          f"gpu={choice.gpu} capability={choice.capability}; {choice.reason}", flush=True)
    if choice.linear_attention is not None:
        print(f"linear-attention: {choice.linear_attention}", flush=True)
    return lm, choice
