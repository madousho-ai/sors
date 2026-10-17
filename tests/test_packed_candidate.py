"""candidate 的打包前向: 前缀 (材料 + 问题) 只算一次, 各选项分支在同一次前向里接着跑, 可反传.

语义与逐选项完整编码 (每个分支 = 前缀 + 这个选项 + 读出 token, 单独跑一遍主干) 相同:
全注意力层里分支只看前缀与自己; DeltaNet 层里分支从前缀末尾的卷积历史与递归状态出发.
DeltaNet 只有 GPU kernel, 这里的测试都要 CUDA.
"""

import random
from pathlib import Path

import pytest
import torch

from test_candidate import candidate_batch, candidate_model, examples
from sors.core.menu import with_partners
from sors.training.loss import training_loss

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="DeltaNet kernels need CUDA")


def _setup(blocks=0, checkpointed=False, seed=0):
    m, tok, d, ids = candidate_model(blocks, "qwen35", trainable="full", checkpointed=checkpointed)
    m.to("cuda")
    exs = with_partners(examples(), random.Random(seed))
    data = {k: v.cuda() for k, v in candidate_batch(tok, d, exs).items()}
    return m, tok, d, data


def _reference_hidden(m, data):
    """逐选项完整编码: 分支的整段 token 单独过主干, 取最后一个 token."""
    valid = data["slot_ids"] >= 0
    ids = data["option_input_ids"][valid]
    mask = data["option_attention_mask"][valid]
    return m._encode(ids, mask)[:, -1].float()


def test_packed_readout_matches_encoding_each_candidate_on_its_own():
    from sors.core.packed import packed_candidate_hidden

    m, _, _, data = _setup()
    m.eval()
    valid = data["slot_ids"] >= 0
    with torch.no_grad():
        want = _reference_hidden(m, data)
        got = packed_candidate_hidden(m.decoder, data, valid).float()
    assert got.shape == want.shape
    torch.testing.assert_close(got, want, atol=2e-3, rtol=2e-3)


def test_packed_gradients_match_encoding_each_candidate_on_its_own(monkeypatch):
    """float64 + 精确的逐步 delta rule: fla 的 float32 kernel 在 dt_bias / A_log 上有百分之几的相对误差
    (逐选项编码时填充与不填充之间就差 4%), 看不出打包本身对不对. 换成精确实现后两条路应当逐位接近."""
    import fla.ops.gated_delta_rule as fla_gdn
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen35
    from exact_gdn import exact_gated_delta_rule
    from sors.core.packed import packed_candidate_hidden

    monkeypatch.setattr(fla_gdn, "chunk_gated_delta_rule", exact_gated_delta_rule)
    monkeypatch.setattr(qwen35, "torch_chunk_gated_delta_rule", exact_gated_delta_rule)
    m, _, _, data = _setup()
    m.double().train()
    valid = data["slot_ids"] >= 0
    ids, mask = data["option_input_ids"][valid], data["option_attention_mask"][valid].bool()
    torch.manual_seed(5)
    weight = torch.randn(int(valid.sum()), m.config.hidden_size, device="cuda", dtype=torch.float64)

    def one_by_one():
        return torch.stack([m._encode(ids[i][mask[i]][None], torch.ones_like(ids[i][mask[i]])[None])[0, -1]
                            for i in range(len(ids))])

    grads = []
    for packed in (False, True):
        m.zero_grad(set_to_none=True)
        h = packed_candidate_hidden(m.decoder, data, valid) if packed else one_by_one()
        (h * weight).sum().backward()
        grads.append({n: p.grad.detach().clone() for n, p in m.named_parameters()
                      if p.grad is not None and n.startswith("base.")})
    assert grads[0].keys() == grads[1].keys() and any("linear_attn" in n for n in grads[0])
    for name, g in grads[0].items():
        error = float((grads[1][name] - g).norm() / g.norm())
        assert error < 1e-5, (name, error)


def test_checkpointed_packed_forward_matches_the_plain_one():
    from sors.core.packed import packed_candidate_hidden

    m, _, _, data = _setup()
    m.train()
    valid = data["slot_ids"] >= 0
    outs = []
    for ckpt in (False, True):
        m.zero_grad(set_to_none=True)
        h = packed_candidate_hidden(m.decoder, data, valid, checkpointed=ckpt)
        h.float().square().sum().backward()
        outs.append((h.detach(), m.option_norm.weight.grad, next(p.grad for n, p in m.named_parameters()
                                                                    if "linear_attn.in_proj_qkv" in n).clone()))
    torch.testing.assert_close(outs[1][0], outs[0][0])
    torch.testing.assert_close(outs[1][2], outs[0][2])


def test_packed_mode_trains_the_backbone_with_the_same_logits_as_full_encoding():
    m, tok, d, data = _setup(blocks=1)
    m.train()
    results = []
    for mode in ("off", "packed"):
        m.set_candidate_prefix_cache(mode)
        m.zero_grad(set_to_none=True)
        logits = m.forward_batch(data)
        loss = training_loss("menu", logits, data["slot_ids"], data["gold"], d, data["target"])
        loss.backward()
        body = sum(float(p.grad.abs().sum()) for n, p in m.named_parameters()
                   if n.startswith("base.") and p.grad is not None)
        results.append((logits.detach(), body))
    valid = torch.isfinite(results[0][0])
    torch.testing.assert_close(results[1][0][valid], results[0][0][valid], atol=3e-3, rtol=3e-3)
    assert results[1][1] > 0


def test_packed_mode_trains_on_a_batch_without_full_option_branches():
    m, tok, d, data = _setup(blocks=1)
    exs = with_partners(examples(), random.Random(0))
    compact = {k: v.cuda() for k, v in candidate_batch(tok, d, exs, option_branches=False).items()}
    m.train()
    m.set_candidate_prefix_cache("packed")
    torch.manual_seed(0)
    full = m.forward_batch(data)
    torch.manual_seed(0)
    logits = m.forward_batch(compact)
    loss = training_loss("menu", logits, compact["slot_ids"], compact["gold"], d, compact["target"])
    loss.backward()
    assert torch.equal(logits, full)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in m.named_parameters() if n.startswith("base."))


def test_packed_mode_is_refused_for_backbones_without_deltanet_support():
    m, tok, d, ids = candidate_model(0, "qwen3", trainable="full")
    m.set_candidate_prefix_cache("packed")
    data = candidate_batch(tok, d)
    with pytest.raises(ValueError, match="packed"):
        m.forward_batch(data)


def test_cli_accepts_packed_prefix_cache_for_candidate():
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_cli", Path(__file__).parents[1] / "scripts/train.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    args = cli.build_parser().parse_args(["--architecture", "candidate", "--candidate-prefix-cache", "packed"])
    assert cli.resolve_candidate_prefix_cache(args, None) == "packed"


def test_candidate_batch_cost_counts_every_branch_for_token_grouping():
    """训练按每行的 token 数分组控显存. candidate 的 attention_mask 只有前缀, 要把各分支 (连同 DeltaNet
    每段一份递归状态的开销) 算进去, 否则 256 个选项的意图题只按前缀长度计, 一组塞进几千个分支."""
    from dataclasses import replace
    from sors.training.loop import batch_costs, BRANCH_STATE_TOKENS

    _, tok, d, _ = candidate_model(0)
    few = examples()[0]
    b = candidate_batch(tok, d, [few])
    prefix = int(b["prefix_lengths"][0])
    suffix = int(b["suffix_attention_mask"][0].sum())
    k = int((b["slot_ids"][0] >= 0).sum())
    assert batch_costs(b, "candidate") == [prefix + suffix + k * BRANCH_STATE_TOKENS]
    assert batch_costs(b, "minimal") == b["attention_mask"].sum(1).tolist()
