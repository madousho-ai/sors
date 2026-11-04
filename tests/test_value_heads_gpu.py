"""扩过 V 头的 Qwen3.5 主干走 fla kernel 与打包前向 (GPU).

V 头多于 k 头时 q、k 要按倍数复制给 V 头; 打包前向 (sors.core.packed) 自己调 fla 的 chunk_gated_delta_rule,
这里确认扩完的主干在这条路上与逐选项完整编码一致, 新头收得到梯度.
"""

import random

import torch

from sors.core.decision import DecisionConfig
from sors.core.model import prepare_model
from sors.core.menu import with_partners
from test_candidate import candidate_batch, examples, tokenizer
from test_decision import tiny_backbone


def _expanded(seed=0):
    tok, d, ids = tokenizer()
    m = prepare_model(tiny_backbone(tok, "qwen35"), ids, None, None, 0., trainable="full", linear_value_heads=4,
                      decision=DecisionConfig(kind="candidate", blocks=0, dim=16, heads=4, option_batch_size=2))
    m.to("cuda")
    exs = with_partners(examples(), random.Random(seed))
    return m, {k: v.cuda() for k, v in candidate_batch(tok, d, exs).items()}


def test_expanded_backbone_packed_readout_matches_encoding_each_candidate_on_its_own():
    from sors.core.packed import packed_candidate_hidden

    m, data = _expanded()
    assert m.decoder.layers[0].linear_attn.num_v_heads == 4
    m.eval()
    valid = data["slot_ids"] >= 0
    with torch.no_grad():
        want = m._encode(data["option_input_ids"][valid], data["option_attention_mask"][valid])[:, -1].float()
        got = packed_candidate_hidden(m.decoder, data, valid).float()
    torch.testing.assert_close(got, want, atol=2e-3, rtol=2e-3)


def test_expanded_backbone_trains_the_new_heads_through_the_packed_path():
    from sors.core.packed import packed_candidate_hidden

    m, data = _expanded()
    m.train()
    valid = data["slot_ids"] >= 0
    packed_candidate_hidden(m.decoder, data, valid).float().pow(2).sum().backward()
    layer = m.decoder.layers[0].linear_attn
    grad = layer.out_proj.weight.grad
    assert torch.isfinite(grad).all()
    new_columns = grad.view(grad.shape[0], 4, -1)[:, 1::2]  # 头 1、3 是新头
    assert new_columns.abs().sum() > 0
