"""Qwen3.5 text backbone uses the common loading and checkpoint interfaces (CPU)."""

import tempfile
from pathlib import Path

import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from _runner import run
from decidophobia.core.attention import load_causal_lm
from decidophobia.core.checkpoint import prepare_from_checkpoint, save_trained
from decidophobia.core.model import last_logits, prepare_model
from decidophobia.training.loop import TrainConfig


def tiny_qwen35():
    torch.manual_seed(11)
    return Qwen3_5ForCausalLM(Qwen3_5TextConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=32, linear_value_head_dim=32,
        layer_types=["linear_attention", "full_attention"],
        tie_word_embeddings=True, use_cache=False,
    ))


def test_qwen35_text_backbone_loads_grows_trains_and_reloads():
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory) / "base"
        tiny_qwen35().save_pretrained(base)
        lm, choice = load_causal_lm(base, device="cpu", dtype=torch.float32, local_files_only=True)
        assert lm.config.model_type == "qwen3_5_text" and choice.backend == "sdpa"
        m = prepare_model(lm, list(range(60, 68)), None, None, 0.0, trainable="full", grad_ckpt=True)
        m.train()
        ids = torch.tensor([[1, 65, 2, 67], [0, 0, 1, 66]])
        mask = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]])
        logits = last_logits(m, ids, mask)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([66, 65]))
        loss.backward()
        grads = {n: p.grad for n, p in m.named_parameters() if p.grad is not None}
        assert all(torch.isfinite(g).all() for g in grads.values())
        for fragment in ("in_proj_qkv", "self_attn.q_proj", ".rows"):
            assert any(fragment in n and g.abs().sum() > 0 for n, g in grads.items()), fragment
        torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=1e-3).step()
        m.eval()
        want = last_logits(m, ids, mask).detach()
        alone = last_logits(m, ids[1:, 2:], mask[1:, 2:])
        torch.testing.assert_close(alone[0], want[1], rtol=1e-4, atol=1e-5)
        checkpoint = Path(directory) / "trained.safetensors"
        save_trained(m, list(range(60, 68)), TrainConfig(), checkpoint)
        fresh, _ = load_causal_lm(base, device="cpu", dtype=torch.float32, local_files_only=True)
        restored, _ = prepare_from_checkpoint(fresh, list(range(60, 68)), checkpoint)
        torch.testing.assert_close(last_logits(restored.eval(), ids, mask), want, rtol=0, atol=0)


if __name__ == "__main__":
    run(globals())
