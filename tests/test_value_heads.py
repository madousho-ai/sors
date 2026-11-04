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


TOKENS = list(range(56, 64))


@pytest.mark.parametrize("trainable", ["full", "attn-mlp-linear"])
def test_checkpoint_records_the_expansion_and_restores_it_on_a_fresh_base(tmp_path, trainable):
    """训练存档只存可训的部分, 基模照 model_id 重新加载; 扩展记在 adapter 里, 还原时先扩再灌权重."""
    from sors.core.checkpoint import checkpoint_adapter, prepare_from_checkpoint, save_trained
    from sors.core.model import adapter_config, last_logits, prepare_model
    from sors.training.loop import TrainConfig

    m = prepare_model(tiny(2), TOKENS, 4, 8, 0.0, trainable=trainable, linear_value_heads=4)
    assert adapter_config(m)["linear_value_heads"] == 4
    ids, mask = IDS, torch.ones_like(IDS)
    m.train()
    last_logits(m, ids, mask)[:, 57].sum().backward()
    torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=1e-2).step()
    m.eval()
    with torch.no_grad():
        want = last_logits(m, ids, mask)
    path = tmp_path / "trained.safetensors"
    save_trained(m, TOKENS, TrainConfig(), path)
    assert checkpoint_adapter(path)["linear_value_heads"] == 4
    restored, _ = prepare_from_checkpoint(tiny(2), TOKENS, path)
    with torch.no_grad():
        torch.testing.assert_close(last_logits(restored.eval(), ids, mask), want, rtol=0, atol=0)


def test_unexpanded_models_keep_their_adapter_record():
    from sors.core.model import adapter_config, prepare_model

    m = prepare_model(tiny(2), TOKENS, None, None, 0.0, trainable="full")
    assert adapter_config(m) == {"trainable": "full", "lora_r": None, "lora_alpha": None}


@pytest.mark.parametrize("trainable", ["decision-only", "full"])
def test_decision_models_record_the_expansion(trainable):
    from sors.core.decision import DecisionConfig
    from sors.core.model import adapter_config, prepare_model

    m = prepare_model(tiny(2), TOKENS, None, None, 0.0, trainable=trainable, linear_value_heads=4,
                      decision=DecisionConfig(kind="candidate", blocks=0, dim=16, heads=4))
    assert adapter_config(m)["linear_value_heads"] == 4


def _train_cli():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parent.parent / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("train_cli_value_heads", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_cli_adds_the_expansion_to_the_adapter_and_the_run_name():
    cli = _train_cli()
    args = cli.build_parser().parse_args(["--trainable", "full", "--linear-value-heads", "32"])
    adapter = cli.resolve_adapter(args)
    assert adapter == {"trainable": "full", "lora_r": None, "lora_alpha": None, "linear_value_heads": 32}
    vars(args).update(adapter)
    assert "-vh32" in cli.run_tag(args)
    plain = cli.build_parser().parse_args(["--trainable", "full"])
    assert "linear_value_heads" not in cli.resolve_adapter(plain)


def test_train_cli_init_inherits_the_expansion_and_rejects_a_different_count(tmp_path):
    from sors.core.checkpoint import save_trained
    from sors.core.model import prepare_model
    from sors.training.loop import TrainConfig

    path = tmp_path / "trained.safetensors"
    m = prepare_model(tiny(2), TOKENS, None, None, 0.0, trainable="full", linear_value_heads=4)
    save_trained(m, TOKENS, TrainConfig(), path)
    cli = _train_cli()
    got = cli.resolve_adapter(cli.build_parser().parse_args(["--init", str(path)]))
    assert got["linear_value_heads"] == 4
    with pytest.raises(SystemExit):
        cli.resolve_adapter(cli.build_parser().parse_args(["--init", str(path), "--linear-value-heads", "8"]))


def test_resume_cli_rebuilds_the_expanded_backbone():
    import argparse
    import importlib.util
    from pathlib import Path

    from sors.core.model import adapter_config, prepare_model

    script = Path(__file__).resolve().parent.parent / "scripts" / "resume.py"
    spec = importlib.util.spec_from_file_location("resume_cli_value_heads", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    saved = prepare_model(tiny(2), TOKENS, None, None, 0.0, trainable="full", linear_value_heads=4)
    args = argparse.Namespace(lora_r=None, lora_alpha=None, lora_dropout=0., trainable="full", grad_ckpt=False,
                              linear_value_heads=4)
    state = {"metadata": {"architecture": {"kind": "slots"}, "adapter": adapter_config(saved)}}
    got = cli.prepare_resume_model(tiny(2), TOKENS, args, state, None, full=True)
    assert got.config.linear_num_value_heads == 4


def test_branch_state_budget_grows_with_the_recurrent_state():
    """token 预算里每个分支按 8 个 token 记 DeltaNet 的递归状态, 这是 16 头 x 128 x 128 推出来的;
    状态更大时按比例放大, 不比它大的 (含 Qwen3 这类没有线性注意力的) 仍记 8."""
    from transformers import Qwen3_5TextConfig

    from sors.training.loop import branch_state_tokens

    def config(heads, dim=128):
        return Qwen3_5TextConfig(linear_num_key_heads=16, linear_num_value_heads=heads,
                                 linear_key_head_dim=dim, linear_value_head_dim=dim)

    assert branch_state_tokens(config(16)) == 8
    assert branch_state_tokens(config(32)) == 16
    assert branch_state_tokens(config(32, 256)) == 64
    assert branch_state_tokens(tiny(2).config) == 8


def test_batch_costs_charge_each_branch_the_given_state_tokens():
    from sors.training.loop import batch_costs

    batch = {"attention_mask": torch.ones(1, 10), "prefix_lengths": torch.tensor([10]),
             "suffix_attention_mask": torch.ones(1, 3, 2), "slot_ids": torch.tensor([[0, 1, 2]])}
    assert batch_costs(batch, "candidate") == [10 + 6 + 3 * 8]
    assert batch_costs(batch, "candidate", state_tokens=16) == [10 + 6 + 3 * 16]


@pytest.mark.parametrize("trainable,decision", [("full", False), ("attn", False), ("decision-only", True)])
def test_training_reads_the_state_budget_from_every_model_wrapper(trainable, decision):
    from sors.core.decision import DecisionConfig
    from sors.core.model import prepare_model
    from sors.training.loop import _backbone_config, branch_state_tokens

    arch = DecisionConfig(kind="candidate", blocks=0, dim=16, heads=4) if decision else None
    m = prepare_model(tiny(2), TOKENS, 4, 8, 0.0, trainable=trainable, linear_value_heads=4, decision=arch)
    assert _backbone_config(m).linear_num_value_heads == 4
    assert branch_state_tokens(_backbone_config(m)) == 8
