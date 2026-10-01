"""Real CUDA forward/backward comparison, including padding, GQA and checkpointing.

Run on the training server:
  PYTHONPATH=src .venv/bin/python tests/test_attention_gpu.py \
    --attn-implementation flash_attention_2 --allow-kernel-download
No pretrained model download is needed. Explicit FlashAttention must really load.
"""

import argparse
import gc
import tempfile

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from decidophobia.core import attention
from decidophobia.core.model import last_logits, prepare_model
from decidophobia.training.loss import slot_cross_entropy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attn-implementation", default="auto", choices=attention.ATTENTION_CHOICES)
    parser.add_argument("--allow-kernel-download", action="store_true")
    args = parser.parse_args()
    assert hasattr(attention, "load_causal_lm"), "the shared model loader is not implemented"
    assert torch.cuda.is_available(), "a CUDA GPU is required"
    with tempfile.TemporaryDirectory(prefix="decidophobia-flash-attention-") as directory:
        cfg = Qwen3Config(vocab_size=256, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                          num_attention_heads=2, num_key_value_heads=1, head_dim=128,
                          attention_dropout=0.0, use_cache=False)
        torch.manual_seed(1)
        source = Qwen3ForCausalLM(cfg)
        source.save_pretrained(directory)
        del source
        ids = torch.tensor([[1, 2, 240, 3, 241, 4, 244], [0, 0, 0, 1, 241, 240, 244]], device="cuda")
        mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1]], device="cuda")
        slots = torch.tensor([[240, 241], [241, 240]], device="cuda")
        gold = torch.tensor([0, 1], device="cuda")
        for trainable, checkpoint in (("full", False), ("full", True), ("attn-mlp", True)):
            results = []
            for backend in ("sdpa", args.attn_implementation):
                torch.manual_seed(2)
                lm, choice = attention.load_causal_lm(
                    directory, attn_implementation=backend, allow_kernel_download=args.allow_kernel_download,
                    local_files_only=True,
                )
                assert lm.config._attn_implementation == choice.implementation
                if backend == "auto" or backend.startswith("flash_attention_"):
                    assert choice.backend.startswith("flash_attention_"), (
                        f"FlashAttention comparison requires a real flash backend: {choice.reason}"
                    )
                model = prepare_model(lm, list(range(240, 256)), 4, 8, 0.0,
                                      trainable=trainable, grad_ckpt=checkpoint)
                model.train()
                logits = last_logits(model, ids, mask)
                loss = slot_cross_entropy(logits, slots, gold)
                loss.backward()
                grads = {n: p.grad.float().cpu().clone() for n, p in model.named_parameters() if p.grad is not None}
                assert grads and all(torch.isfinite(g).all() for g in grads.values())
                assert any(g.abs().max() > 0 for n, g in grads.items() if "q_proj" in n)
                assert any(g.abs().max() > 0 for n, g in grads.items() if n.endswith(".rows"))
                model.eval()
                with torch.no_grad():
                    changed = ids.clone()
                    changed[mask == 0] = 19
                    torch.testing.assert_close(last_logits(model, changed, mask), logits, rtol=0.03, atol=0.02)
                    alone = last_logits(model, ids[1:, 3:], mask[1:, 3:])
                    torch.testing.assert_close(alone[0], logits[1], rtol=0.03, atol=0.02)
                results.append((logits.detach().cpu(), loss.item(), grads, choice))
                del model, lm, logits, loss
                gc.collect()
                torch.cuda.empty_cache()
            reference, candidate = results
            torch.testing.assert_close(candidate[0], reference[0], rtol=0.03, atol=0.02)
            assert abs(candidate[1] - reference[1]) < 0.01
            assert reference[2].keys() == candidate[2].keys()
            delta = sum((candidate[2][n] - g).square().sum() for n, g in reference[2].items()).sqrt()
            scale = sum(g.square().sum() for g in reference[2].values()).sqrt()
            assert delta / scale < 0.06, (delta.item(), scale.item())
            print(f"PASS {trainable=} {checkpoint=} backend={candidate[3].implementation} "
                  f"loss_delta={abs(candidate[1] - reference[1]):.6g} grad_relative_l2={delta / scale:.6g}", flush=True)


if __name__ == "__main__":
    main()
