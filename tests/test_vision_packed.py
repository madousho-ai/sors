"""打包前向读带图的前缀: 前缀算一次、全部分支同一次前向, 结果与逐选项完整编码 (带图) 相同, 可反传.

图在前缀里: 占位符位置换成图像向量, 前缀的旋转位置按 M-RoPE 排, 分支与计算过程从前缀之后的旋转位置接着数.
DeltaNet 只有 GPU kernel, 这里的测试都要 CUDA.
"""

from dataclasses import replace

import pytest
import torch
from test_vision_encoder import pixel_batch, vision_model

from sors.core.vision import encode_images

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="DeltaNet kernels need CUDA")


def _setup(traces=False):
    m, tok, d = vision_model()
    for p in m.base.parameters():
        p.requires_grad_(True)
    m.to("cuda")
    data, _ = pixel_batch(m, tok, d)
    if traces:  # 第 0 行带计算过程, 第 1 行没有
        from test_vision_encoder import pixel_examples

        from sors.core.batch import collate
        exs, _ = pixel_examples()
        exs = [replace(exs[0], trace="red red context"), exs[1]]
        data = collate(exs, tok, d, 4, architecture="candidate", traces=True)
    data = {k: v.cuda() for k, v in data.items()}
    with torch.no_grad():
        data["image_embeds"] = encode_images(m.visual, data["image_pixels"], data["image_grids"])
    return m, data


def _alone(m, data, row, ids):
    """一条完整序列 (前缀 + ...) 单独过主干, 带本行的图, 取最后一个 token."""
    images = (data["image_embeds"][row:row + 1], data["image_grids"][row:row + 1])
    return m._encode(ids[None], torch.ones_like(ids)[None], images)[0, -1]


def _reference(m, data):
    out = []
    for row, col in (data["slot_ids"] >= 0).nonzero().tolist():
        mask = data["option_attention_mask"][row, col].bool()
        out.append(_alone(m, data, row, data["option_input_ids"][row, col][mask]))
    return torch.stack(out)


def test_packed_readout_with_images_matches_encoding_each_candidate_alone():
    from sors.core.packed import packed_candidate_hidden

    m, data = _setup()
    m.eval()
    valid = data["slot_ids"] >= 0
    with torch.no_grad():
        want = _reference(m, data).float()
        got = packed_candidate_hidden(m.decoder, data, valid, image_token_id=m.image_token_id).float()
        flat = packed_candidate_hidden(m.decoder, {k: v for k, v in data.items() if not k.startswith("image_")},
                                       valid).float()
    torch.testing.assert_close(got, want, atol=2e-3, rtol=2e-3)
    assert not torch.allclose(flat, want, atol=1e-2), "the reference does not depend on the images"


def test_teacher_branches_read_the_images_before_their_trace():
    from sors.core.packed import packed_candidate_hidden

    m, data = _setup(traces=True)
    m.eval()
    valid = data["slot_ids"] >= 0
    trace = data["trace_input_ids"][0][data["trace_attention_mask"][0].bool()]
    prefix = data["input_ids"][0][data["attention_mask"][0].bool()][:int(data["prefix_lengths"][0])]
    with torch.no_grad():
        want = []
        for col in range(int(valid[0].sum())):
            suffix = data["suffix_input_ids"][0, col][data["suffix_attention_mask"][0, col].bool()]
            want.append(_alone(m, data, 0, torch.cat([prefix, trace, suffix])))
        _, teacher = packed_candidate_hidden(m.decoder, data, valid, traces=True, image_token_id=m.image_token_id)
    torch.testing.assert_close(teacher.float(), torch.stack(want).float(), atol=2e-3, rtol=2e-3)


def test_packed_gradients_with_images_match_encoding_each_candidate_alone(monkeypatch):
    """float64 + 精确的 delta rule, 理由见 test_packed_candidate 的同名测试."""
    import fla.ops.gated_delta_rule as fla_gdn
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen35
    from exact_gdn import exact_gated_delta_rule

    from sors.core.packed import packed_candidate_hidden

    monkeypatch.setattr(fla_gdn, "chunk_gated_delta_rule", exact_gated_delta_rule)
    monkeypatch.setattr(qwen35, "torch_chunk_gated_delta_rule", exact_gated_delta_rule)
    m, data = _setup()
    m.double().train()
    data["image_embeds"] = data["image_embeds"].double()
    valid = data["slot_ids"] >= 0
    torch.manual_seed(5)
    weight = torch.randn(int(valid.sum()), m.config.hidden_size, device="cuda", dtype=torch.float64)
    grads = []
    for packed in (False, True):
        m.zero_grad(set_to_none=True)
        h = (packed_candidate_hidden(m.decoder, data, valid, image_token_id=m.image_token_id) if packed
             else _reference(m, data))
        (h * weight).sum().backward()
        grads.append({n: p.grad.detach().clone() for n, p in m.named_parameters()
                      if p.grad is not None and n.startswith("base.")})
    assert grads[0].keys() == grads[1].keys() and any("linear_attn" in n for n in grads[0])
    for name, g in grads[0].items():
        error = float((grads[1][name] - g).norm() / g.norm())
        assert error < 1e-5, (name, error)


def test_packed_mode_trains_on_image_batches_with_the_same_logits_as_full_encoding():
    m, data = _setup()
    data.pop("image_embeds")  # forward_batch 自己从像素编码
    m.train()
    results = []
    for mode in ("off", "packed"):
        m.set_candidate_prefix_cache(mode)
        m.zero_grad(set_to_none=True)
        logits = m.forward_batch(data)
        picked = logits.gather(1, data["slot_ids"].clamp_min(0))[data["slot_ids"] >= 0]
        picked.sum().backward()
        grad = next(p.grad for n, p in m.named_parameters() if "linear_attn.in_proj_qkv" in n).clone()
        results.append((picked.detach(), grad))
    torch.testing.assert_close(results[1][0], results[0][0], atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(results[1][1], results[0][1], atol=2e-3, rtol=5e-2)


def test_distillation_encodes_the_pixels_of_image_batches():
    from sors.training.distill import distill_outputs

    m, data = _setup(traces=True)
    want = data.pop("image_embeds")
    m.eval().set_candidate_prefix_cache("packed")
    with torch.no_grad():
        got = distill_outputs(m, data)
        ref = distill_outputs(m, {**data, "image_embeds": want})
    torch.testing.assert_close(got["teacher_logits"], ref["teacher_logits"])
    torch.testing.assert_close(got["student_logits"], ref["student_logits"])
