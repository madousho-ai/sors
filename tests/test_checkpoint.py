"""train.save_trained / load_trained 的测试. 用一个假模型, 不碰 GPU.

跑:  PYTHONPATH=src .venv/bin/python tests/test_checkpoint.py
"""

import tempfile
from dataclasses import asdict

import torch
from torch import nn

from _runner import run
from decidophobia.train import TrainConfig, checkpoint_adapter, load_trained, prepare_from_checkpoint, save_trained


class _Emb(nn.Module):
    def __init__(self, n, h):
        super().__init__()
        self.rows = nn.Parameter(torch.zeros(n, h))


class _Fake(nn.Module):
    """只提供 load_trained 会碰的两样: named_parameters 里的 lora_ 张量, 和 get_input_embeddings().rows."""

    def __init__(self, n_rows, h=4):
        super().__init__()
        self.emb = _Emb(n_rows, h)
        self.lora_A = nn.Parameter(torch.zeros(2, h))

    def get_input_embeddings(self):
        return self.emb


def _save(m, ids):
    f = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    save_trained(m, ids, TrainConfig(), f.name)
    return f.name


TINY_IDS = list(range(40, 64))


def _tiny_lm():
    """一层、hidden 16 的随机 Qwen3, 固定种子: 每次造出来的基模逐位相同. CPU 上一眨眼."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=8)
    torch.manual_seed(0)
    return Qwen3ForCausalLM(cfg)


def _tiny(trainable="attn", r=4, alpha=8):
    """_tiny_lm 走真的 prepare_model (peft LoRA + SlotEmbedding)."""
    from decidophobia.model import prepare_model

    return prepare_model(_tiny_lm(), TINY_IDS, lora_r=r, lora_alpha=alpha, lora_dropout=0.0, trainable=trainable)


def test_checkpoint_records_the_adapter_the_model_was_built_with():
    """评估脚本要先搭一个同形的 LoRA 空壳才能装档, 所以档里得写着 LoRA 挂在哪些层、rank 和 alpha."""
    path = _save(_tiny("attn-mlp", r=4, alpha=12), TINY_IDS)
    assert checkpoint_adapter(path) == {"trainable": "attn-mlp", "lora_r": 4, "lora_alpha": 12}


def test_checkpoint_of_a_d_only_model_records_no_rank():
    path = _save(_tiny("d-only"), TINY_IDS)
    assert checkpoint_adapter(path) == {"trainable": "d-only", "lora_r": None, "lora_alpha": None}


def test_checkpoint_without_an_adapter_record_reads_as_attention_r8_alpha16():
    """记 adapter 之前存的档, 训练时全是 attention LoRA r8 alpha16 (runs/*/result.json 逐个核过)."""
    path = _save(_tiny(), TINY_IDS)
    ck = torch.load(path)
    del ck["adapter"]
    torch.save(ck, path)
    assert checkpoint_adapter(path) == {"trainable": "attn", "lora_r": 8, "lora_alpha": 16}


def test_load_trained_rejects_a_model_whose_alpha_differs_from_the_checkpoint():
    """只差 alpha 时张量形状全对得上, 逐个拷贝不会出错, 但 peft 按 alpha / r 缩放 LoRA 的输出,
    装进去的模型算出来的东西就不是训练时那个. 这种必须拦下."""
    path = _save(_tiny(r=4, alpha=8), TINY_IDS)
    try:
        load_trained(_tiny(r=4, alpha=16), TINY_IDS, path)
    except ValueError:
        return
    raise AssertionError("alpha 16 的模型装进了 alpha 8 的档")


def test_prepare_from_checkpoint_rebuilds_the_trained_model_from_the_file_alone():
    """评估脚本手上只有基模和一个 trained.pt. 照档里的形状搭空壳、装档之后, 输出必须与存档时的模型逐位相同."""
    trained = _tiny("attn-mlp", r=4, alpha=12)
    with torch.no_grad():  # 离开初始化: LoRA B 初值是零, 不动的话 LoRA 形状错了输出也一样
        for n, p in trained.named_parameters():
            if p.requires_grad:
                p.normal_(0, 0.1)
    probe = torch.tensor([[1, 2, 41, 45, 3]])
    trained.eval()
    want = trained(input_ids=probe).logits
    path = _save(trained, TINY_IDS)

    m, cfg = prepare_from_checkpoint(_tiny_lm(), TINY_IDS, path)
    m.eval()
    assert torch.equal(m(input_ids=probe).logits, want)
    assert cfg == asdict(TrainConfig())


def test_load_trained_accepts_checkpoint_with_a_prefix_of_the_model_rows():
    """类型 token 加进来之前存的档只有 256 个 D 行. 那 3 行当时不在提示里、梯度为零,
    留在初始化就是那次训练的真实状态. 所以 ids 是前缀就该装得进去, 多出的行原样不动."""
    old = _Fake(256)
    with torch.no_grad():
        old.emb.rows.fill_(1.0)
        old.lora_A.fill_(2.0)
    path = _save(old, list(range(256)))

    new = _Fake(259)
    with torch.no_grad():
        new.emb.rows.fill_(-1.0)
    load_trained(new, list(range(259)), path)
    assert torch.all(new.emb.rows[:256] == 1.0)
    assert torch.all(new.emb.rows[256:] == -1.0), "档里没有的行必须保持原样"
    assert torch.all(new.lora_A == 2.0)


def test_load_trained_rejects_ids_that_are_not_a_prefix():
    old = _Fake(256)
    path = _save(old, list(range(1, 257)))  # 同样 256 个, 但内容对不上
    new = _Fake(259)
    try:
        load_trained(new, list(range(259)), path)
    except ValueError:
        return
    raise AssertionError("ids 不是前缀却没有报错")


def test_load_trained_rejects_checkpoint_with_more_rows_than_the_model():
    old = _Fake(259)
    path = _save(old, list(range(259)))
    new = _Fake(256)
    try:
        load_trained(new, list(range(256)), path)
    except ValueError:
        return
    raise AssertionError("档比模型多行却没有报错")


if __name__ == "__main__":
    run(globals())
