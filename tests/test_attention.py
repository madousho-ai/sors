"""Attention selection policy: hardware probes are mocked; no model or GPU work."""

import importlib
from contextlib import ExitStack
from unittest.mock import patch

import torch

from _runner import run


def _module():
    try:
        return importlib.import_module("decidophobia.core.attention")
    except ModuleNotFoundError as exc:
        raise AssertionError("the attention selector is not implemented") from exc


def _resolve(capability, available, requested="auto", dtype=torch.bfloat16, **kwargs):
    mod = _module()
    attempted = []

    def probe(backend, allow_kernel_download, *, capability=None):
        attempted.append(backend)
        if backend not in available:
            raise ImportError(f"{backend} unavailable")
        return available[backend]

    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.cuda, "is_available", return_value=True))
        stack.enter_context(patch.object(torch.cuda, "get_device_capability", return_value=capability))
        stack.enter_context(patch.object(torch.cuda, "get_device_name", return_value="test GPU"))
        stack.enter_context(patch.object(mod, "_probe_backend", side_effect=probe))
        choice = mod.resolve_attention(requested, device="cuda:1", dtype=dtype, **kwargs)
    return choice, attempted


def test_ampere_and_ada_choose_fa2_even_when_other_versions_are_installed():
    for capability in ((8, 0), (8, 6), (8, 9)):
        choice, attempted = _resolve(capability, {f"flash_attention_{v}": f"flash_attention_{v}" for v in (2, 3, 4)})
        assert choice.implementation == "flash_attention_2", choice
        assert attempted == ["flash_attention_2"], attempted
        assert choice.device == "cuda:1" and choice.capability == capability


def test_hopper_prefers_fa3_and_can_fall_back_to_installed_fa2():
    choice, _ = _resolve((9, 0), {"flash_attention_3": "flash_attention_3", "flash_attention_2": "flash_attention_2"})
    assert choice.implementation == "flash_attention_3"
    choice, _ = _resolve((9, 0), {"flash_attention_2": "flash_attention_2"})
    assert choice.implementation == "flash_attention_2"
    assert "flash_attention_3" in choice.reason


def test_blackwell_uses_fa4_or_compatible_fa2_and_never_hopper_fa3():
    for capability in ((10, 0), (12, 0)):
        choice, attempted = _resolve(capability, {"flash_attention_4": "flash_attention_4", "flash_attention_3": "flash_attention_3"})
        assert choice.implementation == "flash_attention_4" and attempted == ["flash_attention_4"]
        choice, attempted = _resolve(capability, {"flash_attention_2": "flash_attention_2"})
        assert choice.implementation == "flash_attention_2" and "flash_attention_3" not in attempted


def test_missing_flash_dependencies_fall_back_with_an_explanation():
    choice, _ = _resolve((8, 0), {})
    assert choice.implementation == "sdpa"
    assert "unavailable" in choice.reason and "flash_attention_2" in choice.reason


def test_explicit_flash_requires_compatible_hardware_and_dependency():
    for capability, available, requested in (
        ((8, 0), {"flash_attention_3": "flash_attention_3"}, "flash_attention_3"),
        ((8, 0), {}, "flash_attention_2"),
        ((12, 0), {"flash_attention_3": "flash_attention_3"}, "flash_attention_3"),
    ):
        try:
            _resolve(capability, available, requested)
        except ValueError as exc:
            assert requested in str(exc)
        else:
            raise AssertionError(f"accepted incompatible or absent {requested}")


def test_float32_and_unsupported_architectures_use_sdpa_without_importing_flash():
    for capability, dtype in (((8, 0), torch.float32), ((7, 5), torch.float16), ((13, 0), torch.bfloat16)):
        choice, attempted = _resolve(capability, {}, dtype=dtype)
        assert choice.implementation == "sdpa" and attempted == []


def test_manual_sdpa_and_eager_bypass_optional_imports():
    for requested in ("sdpa", "eager"):
        choice, attempted = _resolve((9, 0), {}, requested)
        assert choice.implementation == requested and attempted == []


def test_cpu_auto_uses_sdpa_and_cpu_flash_is_rejected():
    mod = _module()
    with patch.object(mod, "_probe_backend", side_effect=AssertionError("CPU probed a CUDA kernel")):
        assert mod.resolve_attention(device="cpu", dtype=torch.float32).implementation == "sdpa"
        try:
            mod.resolve_attention("flash_attention_2", device="cpu", dtype=torch.bfloat16)
        except ValueError:
            pass
        else:
            raise AssertionError("CPU accepted flash attention")


def test_invalid_backend_is_rejected():
    try:
        _module().resolve_attention("flash_attention_99", device="cpu", dtype=torch.float32)
    except ValueError:
        return
    raise AssertionError("unknown backend accepted")


def test_nonzero_attention_dropout_keeps_a_backend_that_implements_it():
    choice, attempted = _resolve((9, 0), {"flash_attention_3": "flash_attention_3", "flash_attention_2": "flash_attention_2"},
                                 attention_dropout=0.1)
    assert choice.implementation == "flash_attention_2" and attempted == ["flash_attention_2"]


def test_head_dimensions_outside_flash_support_keep_sdpa():
    choice, attempted = _resolve((8, 0), {"flash_attention_2": "flash_attention_2"}, head_dim=320)
    assert choice.implementation == "sdpa" and attempted == []


def test_optional_hub_kernels_require_download_opt_in():
    mod = _module()
    from transformers import utils
    from transformers import modeling_flash_attention_utils as flash

    def available_kernel(implementation):
        if implementation != "kernels-community/flash-attn2":
            raise AssertionError(f"unexpected repository {implementation}")

    with patch.object(utils, "is_flash_attn_2_available", return_value=False), \
         patch.object(utils, "is_kernels_available", return_value=True), \
         patch.object(flash, "_lazy_imports", side_effect=available_kernel):
        try:
            mod._probe_backend("flash_attention_2", False)
        except ImportError:
            pass
        else:
            raise AssertionError("Hub kernel loaded without opt-in")
        assert mod._probe_backend("flash_attention_2", True) == "kernels-community/flash-attn2"


def test_native_binary_import_errors_are_explained_by_auto_fallback():
    mod = _module()
    from transformers import utils
    from transformers import modeling_flash_attention_utils as flash

    with patch.object(torch.cuda, "is_available", return_value=True), \
         patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)), \
         patch.object(torch.cuda, "get_device_name", return_value="A100"), \
         patch.object(utils, "is_flash_attn_2_available", return_value=True), \
         patch.object(flash, "_lazy_imports", side_effect=OSError("extension ABI mismatch")):
        choice = mod.resolve_attention(device="cuda")
        assert choice.implementation == "sdpa" and "ABI mismatch" in choice.reason


def test_failed_probe_does_not_poison_transformers_attention_cache():
    mod = _module()
    from transformers import utils
    from transformers import modeling_flash_attention_utils as flash
    with patch.object(utils, "is_flash_attn_2_available", return_value=True), \
         patch.object(flash, "_loaded_implementation", None), \
         patch.object(flash, "_lazy_imports", side_effect=OSError("broken extension")):
        for _ in range(2):
            try:
                mod._probe_backend("flash_attention_2", False)
            except OSError:
                pass
            else:
                raise AssertionError("repeated probe accepted a broken extension")
            assert flash._loaded_implementation is None, "failed probe poisoned the framework cache"


def test_sm120_rejects_hub_fa4_and_native_versions_before_sm120_support():
    mod = _module()
    from transformers import utils
    from transformers import modeling_flash_attention_utils as flash
    for installed, version in ((False, "4.0.0b33"), (True, "4.0.0b30")):
        with patch.object(utils, "is_flash_attn_4_available", return_value=installed), \
             patch.object(utils, "is_kernels_available", return_value=True), \
             patch("importlib.metadata.version", return_value=version), \
             patch.object(flash, "_lazy_imports", side_effect=AssertionError("unsupported SM120 kernel loaded")):
            try:
                mod._probe_backend("flash_attention_4", True, capability=(12, 0))
            except ImportError:
                continue
            raise AssertionError("unsupported SM120 FlashAttention accepted")


def test_cuda_runtime_failures_are_propagated():
    mod = _module()
    with patch.object(torch.cuda, "is_available", return_value=True), \
         patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)), \
         patch.object(torch.cuda, "get_device_name", return_value="A100"), \
         patch.object(mod, "_probe_backend", side_effect=torch.cuda.OutOfMemoryError("test OOM")):
        try:
            mod.resolve_attention()
        except torch.cuda.OutOfMemoryError:
            return
    raise AssertionError("CUDA OOM was silently swallowed")


def test_both_clis_accept_attention_selection_and_explicit_kernel_downloads():
    import pathlib
    for script, base in (("train", []), ("resume", ["--run", "runs/example"])):
        path = pathlib.Path(__file__).parents[1] / "scripts" / f"{script}.py"
        spec = importlib.util.spec_from_file_location(f"attention_{script}_cli", path)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        assert hasattr(cli, "build_parser"), f"{script} needs a testable CLI parser"
        parser = cli.build_parser()
        try:
            args = parser.parse_args([*base, "--attn-implementation", "flash_attention_2", "--allow-kernel-download"])
        except SystemExit as exc:
            raise AssertionError(f"{script} rejected attention flags") from exc
        assert args.attn_implementation == "flash_attention_2" and args.allow_kernel_download


if __name__ == "__main__":
    run(globals())
