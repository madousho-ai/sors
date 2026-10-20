"""两阶段 D1/D2 读出 (--decision-readout dual):
  第一阶段 (前 head_start 步): 每个选项分支在 "Answer:" 位置走主干原来的 lm_head, 整个词表上的交叉熵,
    正确选项目标 <|D1|>, 其余 <|D2|> (软标签按目标概率 t 混合); 每道题正、负两部分各占一半. 不算选项间 softmax.
    此时输出的分数是 log P(D1) - log P(D2).
  第二阶段: 标量头 scorer 直接读 H (不过 option_norm), 初值 e_D1 - e_D2, 与第一阶段的分数逐位同一个式子;
    损失 = 选项间 softmax 交叉熵 + branch_weight × 第一阶段那一项.
菜单每个选项前放 <|D1|>, 与 token 读出相同.
"""

import importlib.util
import pathlib

import pytest
import torch

from test_candidate import candidate_batch
from test_decision import examples, tiny_backbone, tokenizer
from sors.core.decision import DecisionConfig, option_marker
from sors.core.model import prepare_model
from sors.core.tokens import D_TOKENS


def dual_model():
    tok, d, ids = tokenizer()
    m = prepare_model(tiny_backbone(tok), ids, 4, 8, 0., trainable="full",
                      decision=DecisionConfig(kind="candidate", blocks=0, readout="dual"))
    m.set_candidate_prefix_cache("off")
    return m, tok, d


def _branch_logp(m, tok, batch):
    """逐个分支单独编码, 整个词表上的 (log P(D1), log P(D2)), 按 valid.nonzero() 的顺序."""
    d1, d2 = tok.convert_tokens_to_ids(D_TOKENS[1]), tok.convert_tokens_to_ids(D_TOKENS[2])
    out = []
    for r, c in (batch["slot_ids"] >= 0).nonzero().tolist():
        ids = batch["option_input_ids"][r, c][batch["option_attention_mask"][r, c].bool()][None]
        logp = m.base.lm_head(m._encode(ids, torch.ones_like(ids))[0, -1][None]).float().log_softmax(-1)[0]
        out.append((logp[d1], logp[d2]))
    return out


def _scores(logits, batch):
    return torch.stack([logits[r, batch["slot_ids"][r, c]] for r, c in (batch["slot_ids"] >= 0).nonzero().tolist()])


def test_stage_one_scores_are_log_odds_and_stage_two_starts_from_the_same_numbers():
    m, tok, d = dual_model()
    batch = candidate_batch(tok, d, option_marker=True)
    m.eval()
    m.head_active = False
    with torch.no_grad():
        want = torch.stack([a - b for a, b in _branch_logp(m, tok, batch)])
        stage_one = _scores(m.forward_batch(batch), batch)
        m.activate_head()
        stage_two = _scores(m.forward_batch(batch), batch)
    torch.testing.assert_close(stage_one, want, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(stage_two, want, atol=1e-4, rtol=1e-4)
    rows = m.get_input_embeddings().rows
    torch.testing.assert_close(m.scorer.weight[0], (rows[1] - rows[2]).float())


def test_branch_loss_balances_the_gold_branch_against_the_wrong_ones():
    from sors.training.loss import branch_ce
    lp1 = torch.log(torch.tensor([[0.8, 0.1, 0.2, 0.0], [0.5, 0.6, 0.0, 0.0]]).clamp_min(1e-9))
    lp2 = torch.log(torch.tensor([[0.1, 0.7, 0.6, 0.0], [0.4, 0.3, 0.0, 0.0]]).clamp_min(1e-9))
    target = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.2, 0.8, 0.0, 0.0]])
    valid = torch.tensor([[True, True, True, False], [True, True, False, False]])
    q0 = 0.5 * (-lp1[0, 0]) + 0.5 * (-(lp2[0, 1] + lp2[0, 2]) / 2)
    pos1 = -(0.2 * lp1[1, 0] + 0.8 * lp1[1, 1]) / 1.0
    neg1 = -(0.8 * lp2[1, 0] + 0.2 * lp2[1, 1]) / 1.0
    want = (q0 + 0.5 * pos1 + 0.5 * neg1) / 2
    torch.testing.assert_close(branch_ce(lp1, lp2, target, valid), want)


def test_dual_model_reads_marked_menus_and_keeps_its_head_but_not_option_norm_trainable():
    m, *_ = dual_model()
    assert option_marker(m) is True
    assert m.scorer.weight.requires_grad and not m.option_norm.weight.requires_grad
    with pytest.raises(ValueError):
        DecisionConfig(kind="candidate", blocks=1, readout="dual")


def test_stage_one_trains_only_on_the_branch_loss():
    from sors.training.loop import TrainConfig, backward_groups
    from sors.training.loss import branch_ce
    m, tok, d = dual_model()
    m.head_active = False
    exs = examples()
    batch = candidate_batch(tok, d, exs, option_marker=True)
    cfg = TrainConfig(loss="menu", accumulate_gradients=True, micro_batches=1, head_start=1)
    m.train()
    stats = {}
    backward_groups(m, cfg, exs, batch, d, stats=stats)
    got = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad(set_to_none=True)
    m.forward_batch(batch)
    want_loss = branch_ce(*m.pop_branch(), batch["target"], batch["slot_ids"] >= 0)
    want_loss.backward()
    assert abs(stats["branch"] - float(want_loss)) < 1e-5
    for n, p in m.named_parameters():
        if p.grad is not None:
            torch.testing.assert_close(got[n], p.grad, atol=1e-6, rtol=1e-5)
    assert "scorer.weight" not in got


def test_training_switches_to_the_head_at_head_start_and_initialises_it_from_d1_minus_d2():
    from sors.training.loop import TrainConfig, train
    m, tok, d = dual_model()
    rows0 = m.get_input_embeddings().rows.detach().clone()
    body = next(p for n, p in m.named_parameters() if n.startswith("base.") and p.requires_grad
                and not n.endswith(".rows"))
    body0 = body.detach().clone()
    cfg = TrainConfig(steps=3, batch_size=2, k_max=4, loss="menu", log_every=10, eval_every=10, lr_lora=1e-3,
                      lr_embed=0.0, lr_decision=0.0, accumulate_gradients=True, candidate_prefix_cache="off",
                      head_start=2)
    train(m, tok, d, lambda n, rng: examples()[:n], {}, cfg)
    assert m.head_active
    torch.testing.assert_close(m.scorer.weight[0], (rows0[1] - rows0[2]).float())
    assert not torch.equal(body.detach(), body0)


def _cli():
    script = pathlib.Path(__file__).resolve().parents[1] / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("train_cli_dual", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def test_cli_parses_the_two_stage_readout():
    cli = _cli()
    base = ["--architecture", "candidate", "--decision-blocks", "0", "--decision-readout", "dual", "--steps", "3300"]
    args = cli.build_parser().parse_args(base + ["--head-start", "1650", "--branch-weight", "0.5"])
    assert cli.resolve_architecture(args)["readout"] == "dual"
    assert cli.resolve_head_start(args) == 1650 and args.branch_weight == 0.5
    assert cli.train_slot_rows(args) is True
    assert "-hs1650" in cli.run_tag(args) and "-bw0.5" in cli.run_tag(args)
    for bad in (["--head-start", "0"], ["--head-start", "3300"], []):
        with pytest.raises(SystemExit):
            cli.resolve_head_start(cli.build_parser().parse_args(base + bad))
    with pytest.raises(SystemExit):
        cli.resolve_head_start(cli.build_parser().parse_args(
            ["--architecture", "candidate", "--decision-blocks", "0", "--head-start", "10"]))


def test_d1_d2_log_probs_are_computed_in_fp32_even_on_a_bf16_backbone():
    """bf16 的整词表 logits 精度约 0.05, 两个 logit 相减后接近的选项会因批的排法不同而翻转 (v73 第一阶段一致率 .95)."""
    m, tok, d = dual_model()
    raw = m.base
    raw.lm_head.base.to(torch.bfloat16)
    m.eval()
    h = torch.randn(5, raw.config.hidden_size) * 3
    with torch.no_grad():
        lp1, lp2 = m._d1_d2_logp(h)
        w = raw.lm_head.base.weight.float().clone()
        rows = m.get_input_embeddings().rows.float()
        ids = m.get_input_embeddings().ids
        w[ids] = rows
        logp = (h.float() @ w.T).log_softmax(-1)
    torch.testing.assert_close(lp1, logp[:, int(ids[1])], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(lp2, logp[:, int(ids[2])], atol=1e-5, rtol=1e-5)


def test_the_step_zero_evaluation_reads_stage_one_scores():
    from sors.training.loop import TrainConfig, train
    m, tok, d = dual_model()
    seen = []
    real = m.forward_batch
    def spy(batch, **kw):
        seen.append(m.head_active)
        return real(batch, **kw)
    m.forward_batch = spy
    from sors.evaluation.scoring import EvalSet
    es = {"probe": EvalSet(examples(), 2)}
    cfg = TrainConfig(steps=2, batch_size=2, k_max=4, loss="menu", log_every=10, eval_every=10, lr_lora=1e-3,
                      accumulate_gradients=True, candidate_prefix_cache="off", head_start=1)
    train(m, tok, d, lambda n, rng: examples()[:n], es, cfg)
    assert seen and seen[0] is False


def test_fp32_log_probs_backpropagate_to_h_and_the_d_rows():
    m, tok, d = dual_model()
    m.base.lm_head.base.to(torch.bfloat16)
    m.train()
    h = (torch.randn(3, m.base.config.hidden_size) * 3).requires_grad_(True)
    lp1, lp2 = m._d1_d2_logp(h)
    (lp1 - lp2).sum().backward()
    assert h.grad is not None and h.grad.abs().sum() > 0
    assert m.get_input_embeddings().rows.grad.abs().sum() > 0
