"""全局梯度裁剪: --max-grad-norm 把一步里全部可训参数的梯度合起来的范数压到上限, 并记录裁剪前的范数."""

import math

import torch

from test_decision import examples, tiny_model


def _params(*grads):
    ps = []
    for g in grads:
        p = torch.nn.Parameter(torch.zeros(len(g)))
        p.grad = torch.tensor(g, dtype=torch.float32)
        ps.append(p)
    return ps


def test_clipping_scales_all_gradients_together_to_the_limit_and_returns_the_norm_before():
    from sors.training.loop import clip_gradients

    ps = _params([3.0, 0.0], [0.0, 4.0])  # 合起来的范数 5
    before = clip_gradients(ps, 1.0)
    assert math.isclose(before, 5.0, rel_tol=1e-6)
    torch.testing.assert_close(ps[0].grad, torch.tensor([0.6, 0.0]))
    torch.testing.assert_close(ps[1].grad, torch.tensor([0.0, 0.8]))


def test_without_a_limit_gradients_are_untouched_but_the_norm_is_still_reported():
    from sors.training.loop import clip_gradients

    ps = _params([3.0, 0.0], [0.0, 4.0])
    assert math.isclose(clip_gradients(ps, None), 5.0, rel_tol=1e-6)
    torch.testing.assert_close(ps[1].grad, torch.tensor([0.0, 4.0]))


def test_gradients_under_the_limit_are_untouched_and_missing_gradients_are_skipped():
    from sors.training.loop import clip_gradients

    ps = _params([0.3, 0.4])
    ps.append(torch.nn.Parameter(torch.zeros(2)))  # 这一步没收到梯度
    assert math.isclose(clip_gradients(ps, 1.0), 0.5, rel_tol=1e-6)
    torch.testing.assert_close(ps[0].grad, torch.tensor([0.3, 0.4]))


class _Writer:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def flush(self):
        pass


def _run(max_grad_norm):
    from sors.training.loop import TrainConfig, train

    m, tok, d, _ = tiny_model("minimal", False, trainable="full")
    w = _Writer()
    train(m, tok, d, lambda n, rng: examples()[:n], {},
          TrainConfig(steps=2, batch_size=2, k_max=4, loss="menu", eval_every=2, log_every=1, lr_lora=1e-3,
                      lr_embed=1e-3, lr_decision=1e-2, max_grad_norm=max_grad_norm), writer=w)
    return m, w


def test_training_logs_the_gradient_norm_before_clipping_every_log_step():
    _, w = _run(1e-6)
    norms = [(s, v) for t, v, s in w.scalars if t == "train/grad_norm"]
    assert [s for s, _ in norms] == [1, 2] and all(v > 1e-6 for _, v in norms), norms


def test_a_tiny_limit_changes_the_update_and_no_limit_keeps_the_old_one():
    """AdamW 第一步按梯度符号走, 看不出裁剪; 第二步动量里存的是裁剪后的梯度, 两种设置走出的权重不同."""
    torch.manual_seed(0)
    plain, _ = _run(None)
    clipped, _ = _run(1e-6)
    a = torch.cat([p.detach().flatten() for n, p in plain.named_parameters() if not n.startswith("base.")])
    b = torch.cat([p.detach().flatten() for n, p in clipped.named_parameters() if not n.startswith("base.")])
    assert not torch.allclose(a, b)
