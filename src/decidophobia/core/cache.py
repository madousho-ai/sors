"""上下文算一次 KV cache, 多个问题各自接在后面.

推理形状: 调用方给一段 context 和 N 个问题. context 前向一次, 得到 28 层的 K/V;
每个问题只喂自己那段 (菜单 + Answer:), 注意力回看 cache 里的 context. N 个问题
之间互不可见, 各自有 256 个槽的预算.

HF 的 DynamicCache 会被 forward 原地追加, 所以每个分支拿的是一份拷贝.
"""

from __future__ import annotations

import copy

import torch
from transformers import DynamicCache


@torch.no_grad()
def prefix_cache(lm, tok, context: str) -> DynamicCache:
    """对 context 前向一次, 返回它的 KV cache (batch=1)."""
    ids = tok(context, return_tensors="pt", add_special_tokens=False)["input_ids"].to(lm.device)
    cache = DynamicCache()
    lm(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    return cache


@torch.no_grad()
def branch_logits(lm, tok, cache: DynamicCache, question: str) -> torch.Tensor:
    """在 cache 的拷贝上接一段 question, 返回最后位置的 (1, V) logits. 传入的 cache 不变."""
    ids = tok(question, return_tensors="pt", add_special_tokens=False)["input_ids"].to(lm.device)
    n_ctx = cache.get_seq_length()
    pos = torch.arange(n_ctx, n_ctx + ids.shape[1], device=lm.device)[None]
    attn = torch.ones(1, n_ctx + ids.shape[1], dtype=torch.long, device=lm.device)
    out = lm(
        input_ids=ids, position_ids=pos, attention_mask=attn,
        past_key_values=copy.deepcopy(cache), use_cache=True, logits_to_keep=1,
    )
    return out.logits[:, -1, :].float()
