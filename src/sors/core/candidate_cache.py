"""Frozen candidate encoding with one configured, immutable prefix per group.

Cache forks include attention KV, convolution history and DeltaNet recurrence.
Suffixes are bucketed by length: padding after a cached prefix advances a hybrid
recurrent state, even when its attention mask is zero.
"""

from __future__ import annotations

import copy
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


@torch.no_grad()
def shared_candidate_hidden(decoder, batch, valid, chunk: int):
    if chunk < 1:
        raise ValueError("candidate chunk size must be positive")
    locations = valid.nonzero().tolist()
    groups = {}
    for row, col in locations:
        prefix = batch["input_ids"][row][batch["attention_mask"][row].bool()]
        length = int(batch["prefix_lengths"][row])
        key = tuple(prefix[:length].tolist())
        groups.setdefault(key, []).append((row, col))
    result = {}
    device = batch["input_ids"].device
    # Process one prefix group at a time, releasing its cache before the next.
    for prefix, rows in groups.items():
        size = len(prefix)
        cache = None
        if size:
            ids = torch.tensor([prefix], dtype=torch.long, device=device)
            cache = decoder(input_ids=ids, attention_mask=torch.ones_like(ids),
                            position_ids=torch.arange(size, device=device)[None],
                            past_key_values=DynamicCache(config=decoder.config), use_cache=True).past_key_values
        buckets = defaultdict(list)
        for row, col in rows:
            length = int(batch["suffix_attention_mask"][row, col].sum())
            if length < 1:
                raise ValueError("candidate continuation needs a readout token")
            buckets[length].append((row, col))
        for length, bucket in sorted(buckets.items()):
            for start in range(0, len(bucket), chunk):
                group = bucket[start:start + chunk]
                ids = torch.stack([batch["suffix_input_ids"][r, c, -length:] for r, c in group])
                branch = fork_cache(cache, len(group), device) if cache is not None else None
                hidden = decoder(input_ids=ids,
                                 attention_mask=torch.ones((len(group), size + length), dtype=torch.long, device=device),
                                 position_ids=torch.arange(size, size + length, device=device)[None].expand(len(group), -1),
                                 past_key_values=branch, use_cache=branch is not None).last_hidden_state[:, -1].clone()
                for i, key in enumerate(group):
                    result[key] = hidden[i]
                del branch
        del cache
    return torch.stack([result[tuple(key)] for key in locations])


def shared_input_tokens(batch) -> int:
    """Physical token count, with identical prefix views counted once per call."""
    prefixes = set()
    for row, length in enumerate(batch["prefix_lengths"].tolist()):
        ids = batch["input_ids"][row][batch["attention_mask"][row].bool()]
        prefixes.add(tuple(ids[:length].tolist()))
    return sum(map(len, prefixes)) + int(batch["suffix_attention_mask"].sum())


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
