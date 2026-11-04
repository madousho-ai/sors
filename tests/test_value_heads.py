"""Qwen3.5 线性注意力 (Gated DeltaNet) 层的 V 头扩展 (CPU).

扩展后每个新 V 头先复制它共用 q、k 的那个旧头, out_proj 里新头那几列置零: 刚扩完的模型与原模型输出一致,
新列一开始就收得到梯度.
"""

import pytest
import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from sors.core.value_heads import expand_linear_value_heads


def tiny(value_heads=2, seed=11):
    torch.manual_seed(seed)
    return Qwen3_5ForCausalLM(Qwen3_5TextConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=value_heads,
        linear_key_head_dim=32, linear_value_head_dim=32,
        layer_types=["linear_attention", "full_attention"],
        tie_word_embeddings=True, use_cache=False,
    )).eval()


IDS = torch.tensor([[1, 5, 9, 2, 7, 3, 11, 4]])


@pytest.fixture(autouse=True)
def exact_delta_rule(monkeypatch):
    """装了 fla 时 transformers 会走 Triton kernel、要占 GPU; 换成纯 torch 的精确实现, 只在 CPU 上算."""
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen35
    from exact_gdn import exact_gated_delta_rule

    monkeypatch.setattr(qwen35, "torch_chunk_gated_delta_rule", exact_gated_delta_rule)


@pytest.mark.parametrize("before,after", [(2, 4), (2, 8), (4, 8)])
def test_expansion_keeps_the_model_output(before, after):
    lm = tiny(before)
    with torch.no_grad():
        want = lm(input_ids=IDS).logits
    expand_linear_value_heads(lm, after)
    layer = lm.model.layers[0].linear_attn
    assert layer.num_v_heads == after and lm.config.linear_num_value_heads == after
    assert layer.out_proj.weight.shape == (64, after * 32)
    assert layer.conv1d.weight.shape[0] == 2 * 64 + after * 32
    with torch.no_grad():
        got = lm(input_ids=IDS).logits
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


def test_new_heads_receive_gradients():
    lm = tiny(2)
    expand_linear_value_heads(lm, 4)
    lm.train()
    lm(input_ids=IDS).logits[0, -1, 3].backward()
    layer = lm.model.layers[0].linear_attn
    # V 头 1、3 是新头 (头 j 共用 k 头 j // 2, 原来的头 i 落在 2i)
    new_columns = layer.out_proj.weight.grad.view(64, 4, 32)[:, 1::2]
    assert new_columns.abs().sum() > 0


def test_expansion_rejects_shrinking_and_uneven_counts():
    for heads in (0, 1, 3):
        with pytest.raises(ValueError):
            expand_linear_value_heads(tiny(2), heads)


def test_expanding_to_the_current_count_changes_nothing():
    lm = tiny(4)
    before = {n: p.clone() for n, p in lm.named_parameters()}
    expand_linear_value_heads(lm, 4)
    for n, p in lm.named_parameters():
        assert torch.equal(p, before[n]), n
