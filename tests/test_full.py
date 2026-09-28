"""--trainable full (主干全参微调) 的测试. 一层、hidden 16 的随机 Qwen3, CPU 上跑.

跑:  PYTHONPATH=src .venv/bin/python tests/test_full.py
"""

import torch

from _runner import run
from decidophobia.core.checkpoint import checkpoint_adapter, prepare_from_checkpoint
from test_checkpoint import TINY_IDS, _perturbed, _save, _tiny, _tiny_lm


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


if __name__ == "__main__":
    run(globals())
