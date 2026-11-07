"""Frozen candidate encoding with one configured, immutable prefix per group.

Cache forks include attention KV, convolution history and DeltaNet recurrence.
Suffixes are bucketed by length: padding after a cached prefix advances a hybrid
recurrent state, even when its attention mask is zero.
"""

from __future__ import annotations

import copy
import hashlib
from collections import defaultdict
from contextlib import contextmanager

import torch
from transformers import DynamicCache


@contextmanager
def frozen_encoder(decoder):
    """Disable encoder stochasticity and gradients, preserving every mode flag."""
    modes = [(module, module.training) for module in decoder.modules()]
    decoder.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training


def fork_cache(cache, count: int, device):
    """Repeated beam indices fork all state types; linear layers lack repeat()."""
    if count < 1:
        raise ValueError("cache fork requires at least one branch")
    branch = copy.deepcopy(cache)
    if count > 1:
        branch.reorder_cache(torch.zeros(count, dtype=torch.long, device=device))
    return branch


def _prefix(batch, row):
    ids = batch["input_ids"][row][batch["attention_mask"][row].bool()]
    return ids[:int(batch["prefix_lengths"][row])]


def prefix_key(batch, row):
    """两行前缀能否共用一份缓存: token 相同, 带图时图也相同 (同样大小的图展开成同一串占位符)."""
    key = tuple(_prefix(batch, row).tolist())
    image = batch.get("image_pixels", batch.get("image_embeds"))  # 有原始像素就认像素, 与编码器的数值误差无关
    if image is not None:
        from sors.core.vision import row_grids
        content = image[row].detach().float().cpu().numpy().tobytes()
        key += (tuple(row_grids(batch["image_grids"][row])), hashlib.sha256(content).hexdigest())
    return key


@torch.no_grad()
def shared_candidate_hidden(decoder, batch, valid, chunk: int, image_token_id: int | None = None):
    if chunk < 1:
        raise ValueError("candidate chunk size must be positive")
    images = "image_embeds" in batch
    locations = valid.nonzero().tolist()
    groups = {}
    for row, col in locations:
        groups.setdefault(prefix_key(batch, row), []).append((row, col))
    result = {}
    device = batch["input_ids"].device
    # Process one prefix group at a time, releasing its cache before the next.
    for rows in groups.values():
        prefix = _prefix(batch, rows[0][0])
        size = len(prefix)
        cache = None
        start = size  # 分支的第一个 token 的旋转位置; 带图时前缀走得比 token 数短
        if size:
            ids = prefix[None].to(device)
            kwargs = {"input_ids": ids, "position_ids": torch.arange(size, device=device)[None]}
            if images:
                from sors.core.vision import image_inputs, rope_positions, row_grids
                r = rows[0][0]
                x, pos = image_inputs(decoder.embed_tokens, ids, torch.ones_like(ids), image_token_id,
                                      batch["image_embeds"][r:r + 1], batch["image_grids"][r:r + 1])
                kwargs = {"inputs_embeds": x, "position_ids": pos}
                start = rope_positions(prefix, row_grids(batch["image_grids"][r]), image_token_id)[1]
            cache = decoder(attention_mask=torch.ones_like(ids), past_key_values=DynamicCache(config=decoder.config),
                            use_cache=True, **kwargs).past_key_values
        buckets = defaultdict(list)
        for row, col in rows:
            length = int(batch["suffix_attention_mask"][row, col].sum())
            if length < 1:
                raise ValueError("candidate continuation needs a readout token")
            buckets[length].append((row, col))
        for length, bucket in sorted(buckets.items()):
            order = torch.arange(size, size + length, device=device)
            if images:  # 第 0 行是序列序号 (掩码与缓存用), 后三行是 M-RoPE 坐标
                order = torch.cat([order[None], torch.arange(start, start + length, device=device).expand(3, -1)])
            for begin in range(0, len(bucket), chunk):
                group = bucket[begin:begin + chunk]
                ids = torch.stack([batch["suffix_input_ids"][r, c, -length:] for r, c in group])
                branch = fork_cache(cache, len(group), device) if cache is not None else None
                hidden = decoder(input_ids=ids,
                                 attention_mask=torch.ones((len(group), size + length), dtype=torch.long, device=device),
                                 position_ids=order[..., None, :].expand(*order.shape[:-1], len(group), -1),
                                 past_key_values=branch, use_cache=branch is not None).last_hidden_state[:, -1].clone()
                for i, key in enumerate(group):
                    result[key] = hidden[i]
                del branch
        del cache
    return torch.stack([result[tuple(key)] for key in locations])


def shared_input_tokens(batch) -> int:
    """Physical token count, with identical prefix views counted once per call."""
    prefixes = {prefix_key(batch, row): int(batch["prefix_lengths"][row]) for row in range(len(batch["prefix_lengths"]))}
    return sum(prefixes.values()) + int(batch["suffix_attention_mask"].sum())


def shared_branch_limit(batch, token_budget: int) -> int:
    """Budget for snapshot + branch caches and the deepcopy/reorder workspace.

    A single branch that exceeds the budget still runs alone, matching the
    existing service policy. Attention activations and other kernel workspaces
    remain outside this KV-token budget.
    """
    prefix = int(batch["prefix_lengths"].max())
    width = int(batch["option_attention_mask"].sum(-1).max())
    limit = max(1, (token_budget - prefix) // width)
    if prefix and limit > 1:
        limit = min(limit, max(1, token_budget // prefix - 2))
    return limit
