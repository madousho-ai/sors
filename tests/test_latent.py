"""隐式推理步 (candidate, 打包前向): 前缀 (材料 + 问题) 编码完, 再接 n 个隐式位置, 各选项分支挂在它们后面.

第 j 个隐式位置的输入向量 = 「前缀 + 前 j-1 个隐式位置」最后一个位置上的最后一层隐藏状态 (final norm 之后),
不经过词表、不采样 (Coconut). 分支只看前缀、隐式位置与自己; 选项之间仍然互相看不见.
语义与逐条完整编码 (前缀 embedding + 隐式向量 + 选项 embedding 一起过主干) 相同.
"""

import random

import pytest
import torch

from test_candidate import candidate_batch, candidate_model, examples
from sors.core.decision import DecisionConfig, architecture_config
from sors.core.menu import with_partners

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="DeltaNet kernels need CUDA")


def _setup(steps=0):
    m, tok, d, _ = candidate_model(0, "qwen35", trainable="full")
    m.to("cuda")
    exs = with_partners(examples(), random.Random(0))
    data = {k: v.cuda() for k, v in candidate_batch(tok, d, exs, option_branches=False).items()}
    return m, data


def _run(decoder, x):
    pos = torch.arange(x.shape[0], device=x.device)[None]
    return decoder(inputs_embeds=x[None], position_ids=pos, use_cache=False).last_hidden_state[0]


def _latent_reference(m, data, n):
    """逐条完整编码: 每行先逐步算出 n 个隐式向量, 再把每个分支单独接上去编码一遍, 取最后一个 token."""
    decoder = m.decoder
    valid = data["slot_ids"] >= 0
    latents = {}
    for r in sorted({r for r, _ in valid.nonzero().tolist()}):
        prefix = data["input_ids"][r][data["attention_mask"][r].bool()][:int(data["prefix_lengths"][r])]
        x = decoder.embed_tokens(prefix)
        for _ in range(n):
            x = torch.cat([x, _run(decoder, x)[-1:].to(x.dtype)])
        latents[r] = x
    out = []
    for r, c in valid.nonzero().tolist():
        k = int(data["suffix_attention_mask"][r, c].sum())
        suffix = decoder.embed_tokens(data["suffix_input_ids"][r, c, -k:])
        out.append(_run(decoder, torch.cat([latents[r], suffix]))[-1])
    return torch.stack(out)


@cuda
@pytest.mark.parametrize("n", [1, 3])
def test_latent_readout_matches_feeding_each_last_hidden_state_back_as_the_next_input(n, monkeypatch):
    """float64 + 精确 delta rule: 小随机模型在 fla 的 float32 kernel 下对漏掉的递归状态不敏感 (差异在 3e-3 以内)."""
    import fla.ops.gated_delta_rule as fla_gdn
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen35
    from exact_gdn import exact_gated_delta_rule
    from sors.core.packed import packed_candidate_hidden

    monkeypatch.setattr(fla_gdn, "chunk_gated_delta_rule", exact_gated_delta_rule)
    monkeypatch.setattr(qwen35, "torch_chunk_gated_delta_rule", exact_gated_delta_rule)
    m, data = _setup()
    m.double().eval()
    valid = data["slot_ids"] >= 0
    with torch.no_grad():
        want = _latent_reference(m, data, n)
        got = packed_candidate_hidden(m.decoder, data, valid, latent_steps=n)
    torch.testing.assert_close(got, want, atol=1e-9, rtol=1e-9)


@cuda
def test_zero_latent_steps_is_the_plain_packed_forward():
    from sors.core.packed import packed_candidate_hidden

    m, data = _setup()
    m.eval()
    valid = data["slot_ids"] >= 0
    with torch.no_grad():
        assert torch.equal(packed_candidate_hidden(m.decoder, data, valid, latent_steps=0),
                           packed_candidate_hidden(m.decoder, data, valid))


@cuda
def test_latent_gradients_flow_through_every_latent_pass_like_the_reference(monkeypatch):
    import fla.ops.gated_delta_rule as fla_gdn
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen35
    from exact_gdn import exact_gated_delta_rule
    from sors.core.packed import packed_candidate_hidden

    monkeypatch.setattr(fla_gdn, "chunk_gated_delta_rule", exact_gated_delta_rule)
    monkeypatch.setattr(qwen35, "torch_chunk_gated_delta_rule", exact_gated_delta_rule)
    m, data = _setup()
    m.double().train()
    valid = data["slot_ids"] >= 0
    grads = []
    for packed in (False, True):
        m.zero_grad(set_to_none=True)
        h = packed_candidate_hidden(m.decoder, data, valid, latent_steps=2) if packed else _latent_reference(m, data, 2)
        torch.manual_seed(5)
        (h * torch.randn_like(h)).sum().backward()
        grads.append({n: p.grad.detach().clone() for n, p in m.named_parameters()
                      if p.grad is not None and n.startswith("base.")})
    assert grads[0].keys() == grads[1].keys()
    for name, g in grads[0].items():
        assert float((grads[1][name] - g).norm() / g.norm()) < 1e-5, name


def test_latent_steps_enter_the_architecture_record_only_when_used():
    plain = DecisionConfig(kind="candidate", blocks=0)
    assert "latent_steps" not in architecture_config(type("M", (), {"decision_config": plain})())
    latent = DecisionConfig(kind="candidate", blocks=0, latent_steps=4)
    assert architecture_config(type("M", (), {"decision_config": latent})())["latent_steps"] == 4
    with pytest.raises(ValueError, match="latent"):
        DecisionConfig(kind="minimal", blocks=1, latent_steps=2)
    with pytest.raises(ValueError, match="latent"):
        DecisionConfig(kind="candidate", blocks=0, latent_steps=-1)


def _latent_model(n):
    from test_decision import tokenizer, tiny_backbone
    from sors.core.model import prepare_model

    tok, d, ids = tokenizer()
    cfg = DecisionConfig(kind="candidate", blocks=0, dim=16, heads=4, option_batch_size=2, latent_steps=n)
    return prepare_model(tiny_backbone(tok, "qwen35"), ids, 4, 8, 0., trainable="full", decision=cfg), tok, d


def test_a_latent_model_only_runs_the_packed_forward():
    m, _, _ = _latent_model(2)
    assert m.candidate_prefix_cache == "packed"
    for mode in ("auto", "on", "off"):
        with pytest.raises(ValueError, match="latent"):
            m.set_candidate_prefix_cache(mode)


@cuda
def test_a_latent_model_scores_options_from_the_latent_readout():
    from sors.core.packed import packed_candidate_hidden

    m, tok, d = _latent_model(2)
    m.to("cuda").eval()
    exs = with_partners(examples(), random.Random(0))
    data = {k: v.cuda() for k, v in candidate_batch(tok, d, exs, option_branches=False).items()}
    valid = data["slot_ids"] >= 0
    with torch.no_grad():
        logits = m.forward_batch(data)
        h = packed_candidate_hidden(m.decoder, data, valid, latent_steps=2).to(m.option_norm.weight.dtype)
        want = m._candidate_output(h, data["slot_ids"])
    torch.testing.assert_close(logits, want)


def test_group_cost_counts_each_latent_position_and_its_recurrent_state():
    """隐式位置逐个接在缓存后面算 (不重算前缀): 每步多 1 个 token, 外加每层一份 DeltaNet 递归状态."""
    from sors.training.loop import BRANCH_STATE_TOKENS, batch_costs

    _, tok, d, _ = candidate_model(0)
    b = candidate_batch(tok, d, [examples()[0]])
    base = batch_costs(b, "candidate")[0]
    assert batch_costs(b, "candidate", latent_steps=3) == [base + 3 * (1 + BRANCH_STATE_TOKENS)]


def _cli():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("train_cli", Path(__file__).parents[1] / "scripts/train.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def test_cli_puts_latent_steps_into_the_candidate_architecture():
    cli = _cli()
    args = cli.build_parser().parse_args(["--architecture", "candidate", "--decision-blocks", "0",
                                          "--candidate-prefix-cache", "packed", "--decision-latent-steps", "4"])
    assert cli.resolve_architecture(args)["latent_steps"] == 4
    plain = cli.build_parser().parse_args(["--architecture", "candidate", "--decision-blocks", "0"])
    assert "latent_steps" not in cli.resolve_architecture(plain)


def test_training_prepares_batch_costs_with_the_latent_positions():
    from sors.training.loop import BRANCH_STATE_TOKENS, _prepare_batch
    from test_decision import tokenizer

    tok, d, _ = tokenizer()
    exs = [examples()[0]]
    spec = (list(d), 4, "context-first", 512, False, False, "candidate", 0.0, False, True, False, False,
            BRANCH_STATE_TOKENS)
    _, _, _, plain = _prepare_batch(spec, (exs, None), tok)
    _, _, _, latent = _prepare_batch(spec + (2,), (exs, None), tok)
    assert latent == [plain[0] + 2 * (1 + BRANCH_STATE_TOKENS)]


@cuda
def test_an_exported_latent_model_reloads_with_its_latent_steps_and_the_same_scores(tmp_path):
    from sors.core import pretrained

    m, tok, d = _latent_model(2)
    m.to("cuda").eval()
    exs = with_partners(examples(), random.Random(0))
    data = {k: v.cuda() for k, v in candidate_batch(tok, d, exs, option_branches=False).items()}
    with torch.inference_mode():
        expected = m.forward_batch(data)
    settings = {"layout": "context-first", "type_marker": False, "context_marker": False,
                "max_length": 8192, "candidate_prefix_cache": "packed"}
    pretrained.save_pretrained(m, tok, settings, tmp_path / "release")
    restored, _, _, _ = pretrained.load_pretrained(tmp_path / "release", device="cuda")
    assert restored.decision_config.latent_steps == 2 and restored.candidate_prefix_cache == "packed"
    with torch.inference_mode():
        torch.testing.assert_close(restored.forward_batch(data), expected)
