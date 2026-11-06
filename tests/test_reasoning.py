"""推理内存模块 (sors.core.reasoning.ReasonMemory), CPU.

前缀 (材料 + 问题) 写进一个 delta-rule 矩阵 M, 草稿位在 M 上读写 T 轮, 选项读出 token 再读 M 加回自己.
输出投影初始为零: 刚接上时对读出向量的修改恰好为零.
"""

import pytest
import torch

from sors.core.reasoning import ReasonMemory

HIDDEN = 32


@pytest.fixture(autouse=True)
def exact_delta_rule(monkeypatch):
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen35
    from exact_gdn import exact_gated_delta_rule

    monkeypatch.setattr(qwen35, "torch_chunk_gated_delta_rule", exact_gated_delta_rule)


def module(seed=0, steps=3):
    torch.manual_seed(seed)
    return ReasonMemory(HIDDEN, heads=2, head_dim=8, slots=3, slot_dim=16, steps=steps, attn_heads=2).double()


def inputs(lengths=(5, 9), seed=1):
    g = torch.Generator().manual_seed(seed)
    prefixes = [torch.randn(n, HIDDEN, generator=g, dtype=torch.float64) for n in lengths]
    readout = torch.randn(5, HIDDEN, generator=g, dtype=torch.float64)
    owner = torch.tensor([0, 0, 1, 1, 1])  # 前两个选项属于第 0 题
    return prefixes, readout, owner


def perturb(m, scale=0.1):
    with torch.no_grad():
        m.out.weight.normal_(0, scale)


def test_zero_output_projection_leaves_the_readout_unchanged():
    prefixes, readout, owner = inputs()
    delta = module()(prefixes, readout, owner)
    assert delta.shape == readout.shape
    assert torch.equal(delta, torch.zeros_like(delta))


def test_output_projection_gets_gradient_from_the_first_step():
    prefixes, readout, owner = inputs()
    m = module()
    m(prefixes, readout, owner).sum().backward()
    assert m.out.weight.grad.abs().sum() > 0


def test_every_parameter_is_on_the_path_once_the_output_moves():
    prefixes, readout, owner = inputs()
    m = module()
    perturb(m)
    m(prefixes, readout, owner).pow(2).sum().backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not dead, dead


def test_each_question_is_independent_of_the_others_in_its_batch():
    prefixes, readout, owner = inputs()
    m = module()
    perturb(m)
    both = m(prefixes, readout, owner)
    alone = m(prefixes[1:], readout[2:], owner[2:] - 1)
    torch.testing.assert_close(both[2:], alone, rtol=1e-10, atol=1e-12)
    short = m(prefixes[:1], readout[:2], owner[:2])  # 批里它被补齐到 9 个位置
    torch.testing.assert_close(both[:2], short, rtol=1e-10, atol=1e-12)


def test_the_readout_depends_on_its_own_prefix_content():
    prefixes, readout, owner = inputs()
    m = module()
    perturb(m)
    before = m(prefixes, readout, owner)
    changed = [prefixes[0], prefixes[1].flip(0)]
    after = m(changed, readout, owner)
    torch.testing.assert_close(after[:2], before[:2], rtol=1e-10, atol=1e-12)
    assert (after[2:] - before[2:]).abs().max() > 1e-6


def test_reasoning_steps_change_the_result():
    prefixes, readout, owner = inputs()
    outs = []
    for steps in (0, 3):
        m = module(steps=steps)
        perturb(m)
        outs.append(m(prefixes, readout, owner))
    assert (outs[0] - outs[1]).abs().max() > 1e-6
