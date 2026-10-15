"""minimal 架构 0 个决策块: 直接拿最后一层 (final norm 之后) 的输出打分.

选项 i 的表示 = 最后一层在它说明末尾 token 上的隐藏状态, 问题表示 = 最后一个 token 上的隐藏状态,
分数 = <option_norm(option_proj(选项)), query(问题)> / sqrt(dim). 没有决策块, 没有残差累加, 没有回写.
"""

import math
import tempfile
from pathlib import Path

import torch

from test_decision import batch, decision_module, examples, tiny_backbone, tokenizer


def zero_block_model(family="qwen3", trainable="full", grad_ckpt=False):
    from sors.core.model import prepare_model

    tok, d, ids = tokenizer()
    config = decision_module().DecisionConfig(kind="minimal", dim=16, heads=4, blocks=0)
    m = prepare_model(tiny_backbone(tok, family), ids, 4, 8, 0.0, trainable=trainable, grad_ckpt=grad_ckpt,
                      decision=config)
    return m, tok, d, ids


def test_zero_blocks_are_accepted_for_minimal_only_without_feedback():
    DecisionConfig = decision_module().DecisionConfig
    assert DecisionConfig(kind="minimal", blocks=0).resolve(4).layers == ()
    for bad in (dict(kind="minimal", blocks=0, feedback=True), dict(kind="structural", blocks=0),
                dict(kind="minimal", blocks=-1)):
        try:
            DecisionConfig(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {bad}")


def test_scores_read_the_last_layer_at_each_option_end_and_at_the_final_token():
    # Qwen3.5 的线性注意力只有 GPU kernel, 没有 GPU 时只测 Qwen3
    for family in ("qwen3", "qwen35") if torch.cuda.is_available() else ("qwen3",):
        m, tok, d, _ = zero_block_model(family)
        dev = "cuda" if family == "qwen35" else "cpu"
        m.to(dev)
        m.eval()
        b = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch(m, tok, d).items()}
        assert len(m.blocks) == 0
        with torch.no_grad():
            z = m.forward_batch(b)
            h = m._encode(b["input_ids"], b["attention_mask"]).float()
            pos = b["option_positions"].clamp_min(0)
            opts = h.gather(1, pos[..., None].expand(-1, -1, h.shape[-1]))
            u = m.option_norm(m.option_proj(opts))
            q = m.query(h[:, -1])
            want = (u * q[:, None]).sum(-1) / math.sqrt(u.shape[-1])
        slots = b["slot_ids"]
        valid = slots >= 0
        got = z.gather(1, slots.clamp_min(0))
        torch.testing.assert_close(got[valid], want[valid], atol=1e-5, rtol=1e-4)
        assert torch.isinf(z).sum(1).tolist() == [z.shape[1] - int(k) for k in valid.sum(1)]


def test_training_reaches_the_backbone_and_both_projections():
    m, tok, d, _ = zero_block_model()
    m.train()
    b = batch(m, tok, d)
    logits = m.forward_batch(b).gather(1, b["slot_ids"].clamp_min(0))
    (-torch.log_softmax(logits[:, :2], -1)[:, 0].mean()).backward()
    assert m.option_proj.weight.grad.abs().sum() > 0 and m.query[1].weight.grad.abs().sum() > 0
    body = [p for n, p in m.base.named_parameters() if p.requires_grad and not n.endswith(".rows")]
    assert body and sum(float(p.grad.abs().sum()) for p in body if p.grad is not None) > 0


def test_checkpointed_forward_matches_the_plain_one():
    plain, tok, d, _ = zero_block_model()
    ckpt, _, _, _ = zero_block_model(grad_ckpt=True)
    ckpt.load_state_dict(plain.state_dict())
    for m in (plain, ckpt):
        m.train()
    b = batch(plain, tok, d)
    a = plain.forward_batch(b)
    c = ckpt.forward_batch(b)
    valid = torch.isfinite(a)
    torch.testing.assert_close(c[valid], a[valid])


def test_saved_weights_reload_into_a_fresh_zero_block_model():
    from sors.core.checkpoint import load_trained, save_trained
    from sors.training.loop import TrainConfig

    m, tok, d, ids = zero_block_model()
    with torch.no_grad():
        m.option_proj.weight.add_(0.5)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trained.safetensors"
        save_trained(m, ids, TrainConfig(loss="menu"), path)
        fresh, *_ = zero_block_model()
        load_trained(fresh, ids, path)
    torch.testing.assert_close(fresh.option_proj.weight, m.option_proj.weight)
    m.eval(), fresh.eval()
    b = batch(m, tok, d)
    with torch.no_grad():
        torch.testing.assert_close(fresh.forward_batch(b), m.forward_batch(b))


def test_cli_accepts_zero_minimal_blocks_without_feedback():
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_cli", Path(__file__).parents[1] / "scripts/train.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    p = cli.build_parser()
    arch = cli.resolve_architecture(p.parse_args(["--architecture", "minimal", "--decision-blocks", "0"]))
    assert arch["blocks"] == 0 and arch["feedback"] is False
    try:
        cli.resolve_architecture(p.parse_args(["--architecture", "minimal", "--decision-blocks", "0",
                                               "--decision-feedback"]))
    except SystemExit:
        pass
    else:
        raise AssertionError("accepted zero blocks with feedback")
