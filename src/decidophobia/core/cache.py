"""上下文算一次 KV cache, 多个问题的分支接在后面.

推理形状: 调用方给一段 context 和 N 个问题. context 前向一次, 得到每层的 K/V (prefix_cache);
N 个问题各自只喂自己那段 (菜单 + Answer:), 注意力回看 cache 里的 context (branch_logits).
N 个分支之间互不可见, 一次前向算完: cache 在 batch 维复制 N 份, 分支左填充到同长,
填充夹在 context 与分支之间, attention mask 挡住它, 真实 token 的位置号从 context 长度接着数 ——
每一行因此等于「context + 这个分支」单独整条前向.

两个函数都吃 token id, 分词由调用方做: 分段编码拼起来要等于整条编码, 分界处 BPE 不能跨界合并
(core.prompt.split_prompt 的两段都以换行收尾, 保证了这一点).
HF 的 DynamicCache 会被 forward 原地追加, 所以分支拿的是一份拷贝, 传入的 cache 不变.
显存: 一次前向的 cache 是 N × (context + 最长分支) 个 token 的 K/V, 调用方按预算把问题分组.
"""

from __future__ import annotations

import copy

import torch
from transformers import DynamicCache


@torch.no_grad()
def prefix_cache(lm, ids: list[int]) -> DynamicCache:
    """对 ids 前向一次, 返回它的 KV cache (batch 1)."""
    if not ids:
        raise ValueError("the prefix has no tokens")
    cache = DynamicCache()
    lm(input_ids=torch.tensor([ids], device=lm.device), past_key_values=cache, use_cache=True, logits_to_keep=1)
    return cache


@torch.no_grad()
def branch_logits(lm, cache: DynamicCache, branches: list[list[int]], pad_id: int) -> torch.Tensor:
    """每个分支接在 cache 后面, 一次前向, 返回各分支最后位置的 logits (N, V), fp32. 传入的 cache 不变."""
    if not branches or not all(branches):
        raise ValueError("need at least one branch, and every branch needs a token")
    n, width, p = len(branches), max(len(b) for b in branches), cache.get_seq_length()
    dev = lm.device
    ids = torch.full((n, width), pad_id, dtype=torch.long)
    pos = torch.full((n, width), p, dtype=torch.long)  # 填充位的位置号无所谓, 它被 mask 挡住
    attn = torch.zeros((n, p + width), dtype=torch.long)
    attn[:, :p] = 1
    for i, b in enumerate(branches):
        pad = width - len(b)
        ids[i, pad:] = torch.tensor(b)
        pos[i, pad:] = torch.arange(p, p + len(b))
        attn[i, p + pad:] = 1
    past = copy.deepcopy(cache)
    past.batch_repeat_interleave(n)
    out = lm(input_ids=ids.to(dev), position_ids=pos.to(dev), attention_mask=attn.to(dev),
             past_key_values=past, use_cache=True, logits_to_keep=1)
    return out.logits[:, -1, :].float()
