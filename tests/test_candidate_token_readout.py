"""candidate 的 token 读出: 每个选项分支在 "Answer:" 位置走主干原来的 lm_head, 分数 = 整个词表上
log P(<|D1|>). 各选项的分数照旧做选项间 softmax 与原来的损失. 另有一项「下一个词只在 {D1, D2} 上」
的约束 (token_mass = -mean log(P(D1) + P(D2))), 让不合适的选项把概率放到 D2 上.
"""

import argparse
import importlib.util
import pathlib
from dataclasses import asdict

import pytest
import torch

from test_candidate import candidate_batch
from test_decision import examples, tiny_backbone, tokenizer
from sors.core.decision import DecisionConfig, architecture_config
from sors.core.model import prepare_model
from sors.core.tokens import D_TOKENS


def token_model(trainable="full"):
    tok, d, ids = tokenizer()
    cfg = DecisionConfig(kind="candidate", blocks=0, readout="token")
    m = prepare_model(tiny_backbone(tok), ids, 4, 8, 0., trainable=trainable, decision=cfg)
    return m, tok, d, ids


def _reference(m, tok, batch):
    """逐个分支单独编码, 取最后一个 token 的 H, 过原来的 lm_head, 整个词表上 log_softmax 取 D1 / D2."""
    d1, d2 = tok.convert_tokens_to_ids(D_TOKENS[1]), tok.convert_tokens_to_ids(D_TOKENS[2])
    valid = batch["slot_ids"] >= 0
    out = []
    for r, c in valid.nonzero().tolist():
        ids = batch["option_input_ids"][r, c][batch["option_attention_mask"][r, c].bool()][None]
        h = m._encode(ids, torch.ones_like(ids))[0, -1]
        logp = m.base.lm_head(h[None]).float().log_softmax(-1)[0]
        out.append((logp[d1], torch.logaddexp(logp[d1], logp[d2])))
    return out


def test_score_is_the_full_vocabulary_log_probability_of_d1():
    m, tok, d, _ = token_model()
    m.set_candidate_prefix_cache("off")
    m.eval()
    batch = candidate_batch(tok, d)
    with torch.no_grad():
        logits = m.forward_batch(batch)
        want = _reference(m, tok, batch)
    valid = batch["slot_ids"] >= 0
    got = [logits[r, batch["slot_ids"][r, c]] for r, c in valid.nonzero().tolist()]
    torch.testing.assert_close(torch.stack(got), torch.stack([w[0] for w in want]), atol=1e-4, rtol=1e-4)


def test_training_forward_exposes_the_d1_d2_mass_term_and_it_reaches_d2():
    m, tok, d, _ = token_model()
    m.set_candidate_prefix_cache("off")
    batch = candidate_batch(tok, d)
    with torch.no_grad():
        m.eval()
        want = -torch.stack([w[1] for w in _reference(m, tok, batch)]).mean()
    m.train()
    m.forward_batch(batch)
    mass = m.pop_token_mass()
    torch.testing.assert_close(mass.detach(), want, atol=1e-4, rtol=1e-4)
    mass.backward()
    rows = m.get_input_embeddings().rows
    d2 = tok.convert_tokens_to_ids(D_TOKENS[2])
    assert rows.grad[m.get_input_embeddings().lut[d2]].abs().sum() > 0
    assert m.pop_token_mass() is None  # 取走一次就清空


def test_eval_forward_leaves_no_mass_term():
    m, tok, d, _ = token_model()
    m.set_candidate_prefix_cache("off")
    m.eval()
    with torch.no_grad():
        m.forward_batch(candidate_batch(tok, d))
    assert m.pop_token_mass() is None


def test_token_readout_needs_candidate_without_set_blocks():
    with pytest.raises(ValueError):
        DecisionConfig(kind="candidate", blocks=1, readout="token")
    with pytest.raises(ValueError):
        DecisionConfig(kind="minimal", blocks=1, readout="token")
    with pytest.raises(ValueError):
        DecisionConfig(kind="candidate", blocks=0, readout="logits")


def test_scorer_records_stay_as_before_so_old_run_tags_do_not_change():
    m, *_ = token_model()
    assert architecture_config(m)["readout"] == "token"
    tok, d, ids = tokenizer()
    plain = prepare_model(tiny_backbone(tok), ids, 4, 8, 0., trainable="full",
                          decision=DecisionConfig(kind="candidate", blocks=0))
    assert "readout" not in architecture_config(plain)


def test_unused_scalar_head_is_frozen_in_token_mode():
    m, *_ = token_model()
    assert not m.scorer.weight.requires_grad and not m.option_norm.weight.requires_grad


def _cli():
    script = pathlib.Path(__file__).resolve().parents[1] / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("train_cli_token_readout", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def test_cli_parses_token_readout_and_keeps_the_d_rows_trainable():
    cli = _cli()
    args = cli.build_parser().parse_args(["--architecture", "candidate", "--decision-blocks", "0",
                                          "--decision-readout", "token"])
    assert cli.resolve_architecture(args)["readout"] == "token"
    assert cli.train_slot_rows(args) is True
    plain = cli.build_parser().parse_args(["--architecture", "candidate", "--decision-blocks", "0"])
    assert "readout" not in cli.resolve_architecture(plain)
    assert cli.run_tag(args) != cli.run_tag(plain)


def test_the_mass_term_changes_how_a_training_step_moves_d2():
    from sors.training.loop import TrainConfig, train
    tok, d, _ = tokenizer()
    d2 = tok.convert_tokens_to_ids(D_TOKENS[2])
    moved, rows_after = {}, {}
    for weight in (0.0, 1.0):
        torch.manual_seed(0)
        m, tok, d, _ = token_model()
        emb = m.get_input_embeddings()
        before = emb.rows[emb.lut[d2]].detach().clone()
        cfg = TrainConfig(steps=2, batch_size=2, k_max=4, loss="menu", log_every=10, eval_every=10, lr_lora=1e-3,
                          lr_embed=1e-2, accumulate_gradients=True, candidate_prefix_cache="off",
                          token_mass_weight=weight)
        train(m, tok, d, lambda n, rng: examples()[:n], {}, cfg)
        moved[weight] = not torch.equal(emb.rows[emb.lut[d2]].detach(), before)
        rows_after[weight] = emb.rows[emb.lut[d2]].detach().clone()
    assert moved[0.0] and moved[1.0] and not torch.equal(*rows_after.values())


def test_marked_branches_put_d1_in_front_of_every_option_name():
    from sors.core.decision_batch import candidate_pieces
    from sors.core.prompt import Special
    ex = examples()[0]
    _, plain = candidate_pieces(ex)
    _, marked = candidate_pieces(ex, option_marker=True)
    for p, q, name in zip(plain, marked, ex.option_names):
        assert q == ["Option: ", Special(D_TOKENS[1]), " ", name, "\n\nAnswer:"]
        assert p == ["Option: ", name, "\n\nAnswer:"]


def test_collate_encodes_the_d1_token_in_every_suffix_only_when_asked():
    tok, d, _ = tokenizer()
    d1 = tok.convert_tokens_to_ids(D_TOKENS[1])
    plain = candidate_batch(tok, d)
    marked = candidate_batch(tok, d, option_marker=True)
    valid = marked["slot_ids"] >= 0
    for r, c in valid.nonzero().tolist():
        suffix = marked["suffix_input_ids"][r, c][marked["suffix_attention_mask"][r, c].bool()].tolist()
        assert suffix.count(d1) == 1
        assert d1 not in plain["suffix_input_ids"][r, c].tolist()


def test_the_token_readout_model_says_it_reads_marked_menus():
    from sors.core.decision import option_marker
    m, *_ = token_model()
    assert option_marker(m) is True
    tok, d, ids = tokenizer()
    plain = prepare_model(tiny_backbone(tok), ids, 4, 8, 0., trainable="full",
                          decision=DecisionConfig(kind="candidate", blocks=0))
    assert option_marker(plain) is False


def test_training_batches_and_evaluation_use_the_marked_menu():
    from sors.evaluation.scoring import score_examples
    from sors.training.loop import _prepare_batch
    m, tok, d, _ = token_model()
    m.set_candidate_prefix_cache("off")
    d1 = tok.convert_tokens_to_ids(D_TOKENS[1])
    spec = (list(d), 4, "context-first", 512, False, False, "candidate", 0.0, False, False, False, True)
    b, *_ = _prepare_batch(spec, (examples(), None), tok)
    assert (b["suffix_input_ids"] == d1).any()
    m.eval()
    with torch.no_grad():
        want = m.forward_batch(candidate_batch(tok, d, option_marker=True))
        q = score_examples(m, tok, d, examples(), 2, 4, 512, "context-first")["q"]
    for i, ex in enumerate(examples()):
        slots = candidate_batch(tok, d)["slot_ids"][i][:len(ex.options)]
        torch.testing.assert_close(torch.tensor(q[i][:len(ex.options)]), want[i, slots].softmax(-1), atol=1e-5, rtol=1e-5)


def test_the_server_reads_marked_menus_for_a_token_readout_model():
    from sors.serve.engine import Engine
    m, tok, d, _ = token_model()
    m.eval()
    from test_serve_engine import _questions
    prompts = Engine(m, tok, d).prompts("red context", _questions())
    assert all(D_TOKENS[1] in p[1] for p in prompts.values())
