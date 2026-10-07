"""Qwen3.5 text backbone uses the common loading and checkpoint interfaces (CPU)."""

import tempfile
from pathlib import Path

import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from _runner import run
from sors.core.attention import load_causal_lm
from sors.core.checkpoint import prepare_from_checkpoint, save_trained
from sors.core.model import adapter_config, last_logits, prepare_model
from sors.training.loop import TrainConfig


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


def test_attn_mlp_linear_puts_lora_on_every_layers_attention():
    """attn-mlp 只认标准注意力的 q/k/v/o, Qwen3.5 的线性注意力层因此没有 LoRA; attn-mlp-linear 补上
    这些层的 in_proj_qkv / in_proj_z / out_proj, 两种注意力层都有可训的注意力投影, MLP 照旧."""
    m = prepare_model(tiny_qwen35(), list(range(60, 68)), lora_r=4, lora_alpha=8, lora_dropout=0.0,
                      trainable="attn-mlp-linear")
    lora = {n for n, p in m.named_parameters() if p.requires_grad and "lora_" in n}
    for layer, kind in enumerate(("linear_attn", "self_attn")):
        attn = {n for n in lora if f"layers.{layer}.{kind}." in n}
        mlp = {n for n in lora if f"layers.{layer}.mlp." in n}
        assert attn, f"layer {layer} ({kind}) has no attention LoRA"
        assert mlp, f"layer {layer} has no MLP LoRA"
    for name in ("in_proj_qkv", "in_proj_z", "out_proj"):
        assert any(f"linear_attn.{name}.lora_" in n for n in lora), name
    assert adapter_config(m) == {"trainable": "attn-mlp-linear", "lora_r": 4, "lora_alpha": 8}


if __name__ == "__main__":
    run(globals())
