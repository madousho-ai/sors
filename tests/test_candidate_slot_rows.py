"""candidate 不用 D token: 提示里只有选项原文, D 行 (rows) 收不到梯度. 训练时把它冻住,
不进主权重、不占优化器状态; 只有开了 --type-marker / --context-marker (那几行会出现在提示里) 才放开."""

import argparse
import importlib.util
import pathlib

import torch

from test_candidate import candidate_model, examples
from sors.training.loop import TrainConfig, train


def _rows(m):
    return m.get_input_embeddings().rows


def _train(m, tok, d, **kw):
    cfg = TrainConfig(steps=2, batch_size=2, k_max=4, loss="menu", log_every=10, eval_every=10, lr_lora=1e-3,
                      lr_decision=1e-2, **kw)
    train(m, tok, d, lambda n, rng: examples()[:n], {}, cfg)


def test_frozen_slot_rows_stay_put_while_backbone_and_head_train():
    m, tok, d, _ = candidate_model(0, trainable="full")
    rows0 = _rows(m).detach().clone()
    head0 = m.scorer.weight.detach().clone()
    body = next(p for n, p in m.named_parameters() if n.startswith("base.") and p.requires_grad
                and not n.endswith(".rows"))
    body0 = body.detach().clone()
    _train(m, tok, d, train_slot_rows=False)
    assert not _rows(m).requires_grad
    assert torch.equal(_rows(m).detach(), rows0)
    assert not torch.equal(m.scorer.weight.detach(), head0)
    assert not torch.equal(body.detach(), body0)


def test_slot_rows_stay_trainable_by_default():
    m, tok, d, _ = candidate_model(0, trainable="full")
    _train(m, tok, d)
    assert _rows(m).requires_grad


def _cli():
    script = pathlib.Path(__file__).resolve().parents[1] / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("train_cli_slot_rows", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def test_cli_trains_slot_rows_only_where_the_prompt_uses_them():
    cli = _cli()
    ns = lambda arch, t=False, c=False: argparse.Namespace(architecture=arch, type_marker=t, context_marker=c)
    assert cli.train_slot_rows(ns("candidate")) is False
    assert cli.train_slot_rows(ns("candidate", t=True)) is True
    assert cli.train_slot_rows(ns("candidate", c=True)) is True
    assert cli.train_slot_rows(ns("minimal")) is True
    assert cli.train_slot_rows(ns("slots")) is True
