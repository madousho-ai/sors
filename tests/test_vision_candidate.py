"""candidate 架构读带图的前缀: 普通逐选项编码与共享前缀缓存两条路 (CPU, 两层随机 Qwen3.5).

一张图在 token 序列里是一串占位符, 批里另带两样: image_embeds (B, P, hidden) 是各行前缀里依次出现的
图像向量 (视觉编码器的输出, 这里随机给), image_grids (B, M, 3) 是各行每张图的 patch 网格, 全零行是空位.
参照: 每个选项的完整序列 (前缀 + 选项) 单独过一遍主干, 占位符处换成图像向量, 位置按 M-RoPE 排.
这里拿词表里的 "short" 当占位符, 示例的 context_label 为空, 免得别的文字撞上它.
"""

from dataclasses import replace

import pytest
import torch
from test_candidate import candidate_batch, candidate_model
from test_decision import examples

from sors.core.vision import rope_positions

GRIDS = [[(1, 4, 4)], [(1, 2, 4)]]  # 第 0 行一张 2x2 (4 个占位符), 第 1 行一张 1x2 (2 个)


def image_examples():
    a, b = examples()
    return [replace(a, query="short short short short red context", context_label=""),
            replace(b, query="long short short blue context", context_label="")]


def image_model(family="qwen35"):
    m, tok, d, _ = candidate_model(0, family)
    m.image_token_id = tok.convert_tokens_to_ids("short")
    return m.eval(), tok, d


def with_images(m, data, seed=0):
    g = torch.Generator().manual_seed(seed)
    data = dict(data)
    data["image_embeds"] = torch.randn(2, 4, m.config.hidden_size, generator=g)
    data["image_grids"] = torch.tensor(GRIDS)
    return data


def reference_logits(m, data, mrope=True):
    """逐选项: 完整序列单独过主干, 占位符换成本行的图像向量, 位置按 M-RoPE (mrope=False 时按 token 数)."""
    decoder = m.decoder
    hidden = []
    for row, col in (data["slot_ids"] >= 0).nonzero().tolist():
        ids = data["option_input_ids"][row, col][data["option_attention_mask"][row, col].bool()]
        x = decoder.embed_tokens(ids[None]).clone()
        spots = (ids == m.image_token_id).nonzero().flatten()
        x[0, spots] = data["image_embeds"][row, :len(spots)].to(x.dtype)
        grids = [tuple(g) for g in data["image_grids"][row].tolist() if any(g)]
        pos, _ = rope_positions(ids, grids, m.image_token_id) if mrope else (torch.arange(len(ids)).expand(3, -1), 0)
        position_ids = torch.cat([torch.arange(len(ids))[None], pos])[:, None]
        h = decoder(inputs_embeds=x, attention_mask=torch.ones_like(ids)[None], position_ids=position_ids,
                    use_cache=False).last_hidden_state[0, -1]
        hidden.append(h)
    return m._candidate_output(torch.stack(hidden).to(m.option_norm.weight.dtype), data["slot_ids"])


@pytest.fixture
def setup():
    m, tok, d = image_model()
    data = with_images(m, candidate_batch(tok, d, image_examples()))
    with torch.no_grad():
        want = reference_logits(m, data)
    return m, data, want


def _valid(logits, data):
    """各行菜单上的 logits, 拉平成一列 (菜单外的列是 -inf, 去掉)."""
    return logits.gather(1, data["slot_ids"].clamp_min(0))[data["slot_ids"] >= 0]


def test_the_reference_depends_on_image_vectors_and_rotary_layout(setup):
    """参照本身要对图像向量和三维位置都敏感, 否则下面几条比不出东西."""
    m, data, want = setup
    with torch.no_grad():
        flat = reference_logits(m, data, mrope=False)
        other = reference_logits(m, with_images(m, data, seed=1))
    assert not torch.allclose(_valid(flat, data), _valid(want, data), atol=1e-4)
    assert not torch.allclose(_valid(other, data), _valid(want, data), atol=1e-4)


@pytest.mark.parametrize("mode", ["off", "on"])
def test_candidate_paths_read_images_like_encoding_each_option_alone(setup, mode):
    m, data, want = setup
    m.set_candidate_prefix_cache(mode)
    with torch.no_grad():
        got = m.forward_batch(data)
    torch.testing.assert_close(_valid(got, data), _valid(want, data), atol=1e-5, rtol=1e-4)


def test_shared_prefix_keeps_rows_whose_tokens_match_but_images_differ_apart():
    """两行前缀 token 完全相同 (同一句话、同样大小的图), 图不同: 各自读自己的图."""
    m, tok, d = image_model()
    a = image_examples()[0]
    data = candidate_batch(tok, d, [a, a])
    data["image_embeds"] = torch.randn(2, 4, m.config.hidden_size, generator=torch.Generator().manual_seed(2))
    data["image_grids"] = torch.tensor([[(1, 4, 4)], [(1, 4, 4)]])
    m.set_candidate_prefix_cache("on")
    with torch.no_grad():
        want = reference_logits(m, data)
        got = m.forward_batch(data)
    torch.testing.assert_close(_valid(got, data), _valid(want, data), atol=1e-5, rtol=1e-4)
    assert not torch.allclose(got[0], got[1], atol=1e-4)


def test_shared_prefix_counts_each_row_with_images_on_its_own():
    from sors.core.candidate_cache import shared_input_tokens

    m, tok, d = image_model()
    a = image_examples()[0]
    data = candidate_batch(tok, d, [a, a])
    text_only = shared_input_tokens(data)
    data["image_embeds"] = torch.randn(2, 4, m.config.hidden_size)
    data["image_grids"] = torch.tensor([[(1, 4, 4)], [(1, 4, 4)]])
    assert shared_input_tokens(data) == text_only + int(data["prefix_lengths"][1])


def test_image_batches_need_the_image_token():
    m, tok, d = image_model()
    data = with_images(m, candidate_batch(tok, d, image_examples()))
    m.image_token_id = None
    with pytest.raises(ValueError, match="image"):
        m.forward_batch(data)


@pytest.mark.parametrize("kind", ["minimal", "structural"])
def test_other_decision_architectures_reject_image_batches(kind):
    from test_decision import batch, tiny_model

    m, tok, d, _ = tiny_model(kind, family="qwen35")
    m.image_token_id = tok.convert_tokens_to_ids("short")
    data = with_images(m, batch(m, tok, d, image_examples()))
    with pytest.raises(ValueError, match="image"):
        m.eval().forward_batch(data)
