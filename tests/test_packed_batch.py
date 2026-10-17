"""打包前向训练用的 candidate 批: 不带逐选项完整分支 (option_input_ids / option_attention_mask).

打包前向只读前缀 (input_ids, attention_mask, prefix_lengths) 与选项续写 (suffix_*). 完整分支张量把前缀给每个选项各抄一份,
形状 (B, k_max, 前缀 + 选项), 长材料 + 256 项菜单时一批近 1 GB, 后台进程序列化、共享内存与拷进显存都按它算.
"""

import functools
import random

import torch

from test_candidate import candidate_batch, candidate_model, examples
from sors.core.menu import with_partners
from sors.training.loop import _prepare_batch

FULL_BRANCHES = ("option_input_ids", "option_attention_mask")


def test_a_batch_without_option_branches_keeps_every_other_tensor_unchanged():
    _, tok, d, _ = candidate_model(0)
    exs = with_partners(examples(), random.Random(0))
    full = candidate_batch(tok, d, exs)
    compact = candidate_batch(tok, d, exs, option_branches=False)
    assert not set(FULL_BRANCHES) & set(compact)
    assert set(compact) == set(full) - set(FULL_BRANCHES)
    for key, value in compact.items():
        assert torch.equal(value, full[key]), key


def test_training_prepares_packed_candidate_batches_without_option_branches():
    _, tok, d, _ = candidate_model(0)
    exs = with_partners(examples(), random.Random(0))
    spec = (list(d), 4, "context-first", 512, False, False, "candidate", 0.0, True)
    full, *_ = _prepare_batch(spec, (exs, None), tok)
    packed, *_ = _prepare_batch(spec + (True,), (exs, None), tok)
    assert set(FULL_BRANCHES) <= set(full)
    assert not set(FULL_BRANCHES) & set(packed)
