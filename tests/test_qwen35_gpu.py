"""Qwen3.5 CUDA kernel dispatch, left padding, checkpointed backward and reload.

Run on a training GPU with flash-linear-attention and kernels already installed.
Hub downloads are explicitly allowed here. No pretrained weights are downloaded.
"""

import gc
import tempfile
from pathlib import Path

import torch

from decidophobia.core.attention import load_causal_lm
from decidophobia.core.checkpoint import prepare_from_checkpoint, save_trained
from decidophobia.core.model import last_logits, prepare_model
from decidophobia.training.loop import TrainConfig
from test_qwen35 import tiny_qwen35


def assert_convolution_kernel(model):
    from kernels import get_kernel
    implementation = get_kernel("kernels-community/causal-conv1d", version=2)
    count = 0
    for module in model.modules():
        functions = getattr(module, "_kernel_funcs", {})
        if "causal_conv1d_fn" in functions:
            count += 1
            assert functions["causal_conv1d_fn"].forward.__func__ is implementation.layers.causal_conv1d_fn.forward
    assert count > 0, "the optimized function must be bound to a model layer"


def main():
    assert torch.cuda.is_available(), "a training GPU is required"
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory) / "base"
        tiny_qwen35().save_pretrained(base)
        ids = torch.tensor([[1, 65, 2, 67], [0, 0, 1, 66]], device="cuda")
        mask = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]], device="cuda")
        results = []
        for fast in (False, True):
            lm, choice = load_causal_lm(base, attn_implementation="sdpa", local_files_only=True,
                                        allow_kernel_download=fast)
            assert lm.use_kernels == fast, "Qwen3.5 must activate the opted-in causal convolution kernel"
            torch.manual_seed(22)
            m = prepare_model(lm, list(range(60, 68)), None, None, 0.0, trainable="full", grad_ckpt=True)
            m.train()
            if fast:
                assert_convolution_kernel(m)
            logits = last_logits(m, ids, mask)
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor([66, 65], device="cuda"))
            loss.backward()
            grads = {n: p.grad.float().cpu().clone() for n, p in m.named_parameters() if p.grad is not None}
            assert all(torch.isfinite(g).all() for g in grads.values())
            for fragment in ("in_proj_qkv", "conv1d.weight", "self_attn.q_proj", ".rows"):
                assert any(fragment in n and g.abs().sum() > 0 for n, g in grads.items()), fragment
            m.eval()
            if fast:
                assert_convolution_kernel(m)
            with torch.no_grad():
                changed = ids.clone()
                changed[mask == 0] = 19
                torch.testing.assert_close(last_logits(m, changed, mask), logits, rtol=0.03, atol=0.02)
                torch.testing.assert_close(last_logits(m, ids[1:, 2:], mask[1:, 2:])[0], logits[1],
                                           rtol=0.03, atol=0.02)
            if fast:
                # Both mode switches must retain the selected kernel, including checkpoint recomputation.
                m.train()
                assert_convolution_kernel(m)
                m.zero_grad(set_to_none=True)
                last_logits(m, ids, mask).sum().backward()
                checkpoint = Path(directory) / "trained.safetensors"
                save_trained(m, list(range(60, 68)), TrainConfig(), checkpoint)
                fresh, _ = load_causal_lm(base, attn_implementation="sdpa", local_files_only=True,
                                          allow_kernel_download=True)
                restored, _ = prepare_from_checkpoint(fresh, list(range(60, 68)), checkpoint)
                with torch.no_grad():
                    torch.testing.assert_close(last_logits(restored.eval(), ids, mask), logits, rtol=0, atol=0)
                del fresh, restored
            results.append((logits.detach().cpu(), grads))
            del m, lm, logits, loss
            gc.collect()
            torch.cuda.empty_cache()
            print(f"PASS qwen35 forward/backward/reload {fast=}", flush=True)
        torch.testing.assert_close(results[0][0], results[1][0], rtol=0.03, atol=0.02)
        delta = sum((results[1][1][n] - g).square().sum() for n, g in results[0][1].items()).sqrt()
        scale = sum(g.square().sum() for g in results[0][1].values()).sqrt()
        assert delta / scale < 0.06, (delta.item(), scale.item())
        print(f"PASS qwen35 accelerated/reference gradient relative L2={delta / scale:.6g}", flush=True)


if __name__ == "__main__":
    main()
