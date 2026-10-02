"""--trainable full (主干全参微调) 的测试. 一层、hidden 16 的随机 Qwen3, CPU 上跑.

跑:  PYTHONPATH=src .venv/bin/python tests/test_full.py
"""

import torch
from torch import nn

from _runner import run
from sors.core.checkpoint import checkpoint_adapter, prepare_from_checkpoint
from sors.core.model import prepare_model
from sors.training.loop import Fp32Master, TrainConfig, train
from test_checkpoint import _EX, TINY_IDS, _perturbed, _save, _tiny, _tiny_lm, _Tok


def test_full_trains_every_body_weight_and_only_the_d_rows_of_the_vocab_matrix():
    """full 放开每一层的 attention、MLP、norm 和最后的 norm, 不套 LoRA; 词嵌入与输出层的那张大矩阵照旧冻结,
    只有 D 行 (rows) 可训."""
    m = _tiny("full")
    trained = {n for n, p in m.named_parameters() if p.requires_grad}
    frozen = {n for n, p in m.named_parameters() if not p.requires_grad}
    assert frozen == {"model.embed_tokens.base.weight", "lm_head.base.weight"}, frozen
    assert {"model.embed_tokens.rows", "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.down_proj.weight", "model.layers.0.input_layernorm.weight",
            "model.norm.weight"} <= trained, trained
    assert not any("lora_" in n for n in trained), trained


def test_checkpoint_of_a_full_model_records_full_with_no_rank():
    path = _save(_tiny("full"), TINY_IDS)
    assert checkpoint_adapter(path) == {"trainable": "full", "lora_r": None, "lora_alpha": None}


def test_prepare_from_checkpoint_rebuilds_a_full_model_from_the_file_alone():
    """档里要有主干的全部权重: 照档搭好、装档之后, 输出与存档时的模型逐位相同."""
    trained = _perturbed("full", r=None, alpha=None)
    probe = torch.tensor([[1, 2, 41, 45, 3]])
    want = trained(input_ids=probe).logits
    path = _save(trained, TINY_IDS)

    m, _ = prepare_from_checkpoint(_tiny_lm(), TINY_IDS, path)
    m.eval()
    assert torch.equal(m(input_ids=probe).logits, want)


def test_full_bf16_body_moves_even_when_each_step_is_below_bf16_resolution():
    """全参时主干是 bf16. 1.0 附近 bf16 的间距是 2^-8 (往下) 与 2^-7 (往上), 单步 1e-3 的改动直接写回 bf16
    会被舍入回 1.0, 权重永远不动. 更新要在 fp32 主权重上累积, 几步之后 bf16 那份才跨过一个间距.
    RMSNorm 的权重初值恰好全是 1.0."""
    m = prepare_model(_tiny_lm().to(torch.bfloat16), TINY_IDS, None, None, 0.0, trainable="full")
    norm = m.model.norm.weight
    assert norm.dtype == torch.bfloat16 and torch.all(norm == 1.0)
    train(m, _Tok(), TINY_IDS, lambda n, rng: [_EX] * n, {},
          TrainConfig(steps=8, batch_size=1, k_max=2, lr_lora=1e-3, eval_every=100, log_every=100))
    assert norm.dtype == torch.bfloat16
    assert bool((norm != 1.0).any()), norm


def test_fp32_master_hands_each_step_only_its_own_gradient():
    """模型上那份 bf16 梯度搬走后要清掉, 否则下一次 backward 累加上去, 第二步拿到的是两步之和."""
    p = nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
    master = Fp32Master([p])
    for scale in (1.0, 2.0):
        (p.float() * scale).sum().backward()
        master.pull_grads()
    assert torch.equal(master.params[0].grad, torch.full((3,), 2.0)), master.params[0].grad


def test_fp32_master_leaves_fp32_params_as_they_are():
    """LoRA 权重本来就是 fp32: 优化器拿到的是参数本身, LoRA 的 run 与加主权重之前逐位相同."""
    p = nn.Parameter(torch.ones(3))
    assert Fp32Master([p]).params[0] is p


if __name__ == "__main__":
    run(globals())
