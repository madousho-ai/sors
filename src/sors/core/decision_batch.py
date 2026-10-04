"""Architecture-specific encoding; semantic options stay aligned with targets."""

from __future__ import annotations

import torch

from sors.core.prompt import DEFAULT_QUESTION, Special, _context, _special_id, encode_prompts, prompt_pieces
from sors.core.tokens import CONTEXT_TOKENS


def _pad(sequences, pad):
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, seq in enumerate(sequences):
        ids[i, -len(seq):] = torch.tensor(seq)
        mask[i, -len(seq):] = 1
    return ids, mask


def _kept(ids, limit, bounds, truncate):
    if len(ids) > limit and not truncate:
        raise ValueError(f"input has {len(ids)} tokens, limit is {limit}")
    keep = list(range(len(ids)))
    over = len(ids) - limit
    if over > 0 and bounds is not None:
        a, b = ids.index(bounds[0]) + 1, ids.index(bounds[1])
        keep = keep[:a] + keep[a + min(over, b - a):]
    return keep[-limit:]


def _minimal_plan(ex, tok, d_ids, layout, type_marker, context_marker, texts, specials):
    """Template steps for one prompt; text segments are appended to texts for one batched encode.

    A text segment that follows an option's code belongs to that option. Steps are
    ("id", token, owner) or ("text", index into texts, owner).
    """
    by_token = {d_ids[c]: j for j, c in enumerate(ex.slot_codes)}
    plan, buf, owner = [], [], None

    def flush():
        if buf:
            plan.append(("text", len(texts), owner))
            texts.append("".join(buf))
            buf.clear()

    for piece in sum(prompt_pieces(ex, layout, type_marker, context_marker), []):
        if isinstance(piece, Special):
            flush()
            if piece not in specials:
                specials[piece] = _special_id(tok, piece)
            owner = by_token.get(specials[piece])
            plan.append(("id", specials[piece], owner))
        else:
            buf.append(piece)
    flush()
    return plan


def _minimal_assemble(ex, plan, enc, limit, bounds, truncate):
    ids, ends, starts = [], {}, {}
    for kind, value, owner in plan:
        if kind == "id":
            if owner is not None:
                starts[owner] = len(ids)
            ids.append(value)
            continue
        if owner is not None:
            boundary = len(f". {ex.option_names[owner]}")
            content = [i for i, (a, b) in enumerate(enc["offset_mapping"][value]) if b > 0 and a < boundary]
            if not content:
                raise ValueError("option has no encoded content")
            ends[owner] = len(ids) + content[-1]
        ids.extend(enc["input_ids"][value])
    keep = _kept(ids, limit, bounds, truncate)
    remap = {old: new for new, old in enumerate(keep)}
    for j in range(len(ex.options)):
        if any(i not in remap for i in range(starts[j], ends[j] + 1)):
            raise ValueError("truncation would remove an option; increase max_length")
    return [ids[i] for i in keep], [remap[ends[j]] for j in range(len(ex.options))]


def _minimal_batch(examples, tok, d_ids, layout, type_marker, context_marker, limit, truncate):
    """Encode every text segment of the batch in one tokenizer call.

    Segments are split at the same template tokens as before, so each segment's
    ids and offsets equal a standalone encode of that segment.
    """
    texts, specials = [], {}
    plans = [_minimal_plan(ex, tok, d_ids, layout, type_marker, context_marker, texts, specials)
             for ex in examples]
    enc = (tok(texts, add_special_tokens=False, split_special_tokens=True, return_offsets_mapping=True)
           if texts else {"input_ids": [], "offset_mapping": []})
    bounds = tuple(tok.convert_tokens_to_ids(x) for x in CONTEXT_TOKENS) if context_marker else None
    return [_minimal_assemble(ex, plan, enc, limit, bounds, truncate) for ex, plan in zip(examples, plans)]


def structural_pieces(ex, type_marker=False, context_marker=False):
    label = ["Question (", Special(f"<|{ex.qtype}|>"), "):"] if type_marker else ["Question:"]
    question = [*label, f" {ex.question if ex.question is not None else DEFAULT_QUESTION}"]
    memory = [*_context(ex, context_marker), "\n\n", *question, "\n\nAnswer:"]
    options = [[*question, "\nOption: ", name] for name in ex.option_names]
    return memory, options


def candidate_pieces(ex, type_marker=False, context_marker=False):
    """A shared textual prefix plus suffixes, each ending at a pretrained readout token.

    Callers encode prefix + suffix together. The prefix tensor in the batch is
    scheduling metadata only; the model forwards the complete candidate branches.
    """
    label = ["Question (", Special(f"<|{ex.qtype}|>"), "):"] if type_marker else ["Question:"]
    question = ex.question if ex.question is not None else DEFAULT_QUESTION
    prefix = [*_context(ex, context_marker), "\n\n", *label, f" {question}\n\n"]
    return prefix, [["Option: ", name, "\n\nAnswer:"] for name in ex.option_names]


def _candidate_suffixes(prefixes, branches):
    """Split token sequences, preserving BPE merges at the textual boundary.

    A standalone prefix's final token can merge with the continuation. Shorten
    to the shared token prefix and leave the merged token in each suffix. Always
    reserve a readout token in the suffix, including degenerate tokenizers.
    """
    lengths, suffixes = [], []
    for prefix, group in zip(prefixes, branches):
        if any(not seq for seq in group):
            raise ValueError("candidate branch has no tokens")
        length = min(len(prefix), min(len(seq) - 1 for seq in group))
        for seq in group:
            if seq[:length] != prefix[:length]:
                length = next(i for i in range(length) if seq[i] != prefix[i])
        lengths.append(length)
        suffixes.extend(seq[length:] for seq in group)
    return lengths, suffixes


def collate_decisions(examples, tokenizer, d_ids, k_max, layout, max_length, type_marker, context_marker,
                      architecture, *, truncate=True):
    from sors.core.batch import targets

    if max_length < 1 or not examples or any(not e.options or len(e.options) > k_max for e in examples):
        raise ValueError("decision batch needs nonempty menus within k_max and positive max_length")
    if any(not n.strip() for ex in examples for n in ex.option_names):
        raise ValueError("decision option descriptions must be nonempty")
    out = targets(examples, d_ids, k_max)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    if architecture == "minimal":
        encoded = _minimal_batch(examples, tokenizer, d_ids, layout, type_marker, context_marker, max_length, truncate)
        ids, mask = _pad([seq for seq, _ in encoded], pad)
        positions = torch.full(out["slot_ids"].shape, -1, dtype=torch.long)
        for i, (seq, ends) in enumerate(encoded):
            positions[i, :len(ends)] = torch.tensor(ends) + ids.shape[1] - len(seq)
        out.update(input_ids=ids, attention_mask=mask, option_positions=positions)
        return out
    if architecture not in ("structural", "candidate") or layout != "context-first":
        raise ValueError(f"{architecture} architecture requires context-first layout")
    if architecture == "candidate":
        memory, suffixes = zip(*(candidate_pieces(e, type_marker, context_marker) for e in examples))
        options = [[prefix + suffix for suffix in group] for prefix, group in zip(memory, suffixes)]
    else:
        memory, options = zip(*(structural_pieces(e, type_marker, context_marker) for e in examples))
    enc = encode_prompts(tokenizer, list(memory) + [p for group in options for p in group])
    bounds = tuple(tokenizer.convert_tokens_to_ids(x) for x in CONTEXT_TOKENS) if context_marker else None
    memories = (enc[:len(examples)] if architecture == "candidate" else
                [[s[j] for j in _kept(s, max_length, bounds, truncate)] for s in enc[:len(examples)]])
    # Keep each complete question/option branch. Silent option truncation would
    # change the candidate's semantics, even though row equivariance still held.
    opt = enc[len(examples):]
    if any(len(s) > max_length for s in opt):
        raise ValueError(f"{architecture} option branch exceeds max_length; increase the limit to preserve full input")
    ids, mask = _pad(memories, pad)
    oi, om = _pad(opt, pad)
    option_ids = torch.full((*out["slot_ids"].shape, oi.shape[1]), pad, dtype=torch.long)
    option_mask = torch.zeros_like(option_ids)
    valid = out["slot_ids"] >= 0
    option_ids[valid], option_mask[valid] = oi, om
    out.update(input_ids=ids, attention_mask=mask, option_input_ids=option_ids, option_attention_mask=option_mask)
    if architecture == "candidate":
        groups, start = [], 0
        for ex in examples:
            groups.append(opt[start:start + len(ex.options)])
            start += len(ex.options)
        lengths, suffixes = _candidate_suffixes(memories, groups)
        si, sm = _pad(suffixes, pad)
        suffix_ids = torch.full((*out["slot_ids"].shape, si.shape[1]), pad, dtype=torch.long)
        suffix_mask = torch.zeros_like(suffix_ids)
        suffix_ids[valid], suffix_mask[valid] = si, sm
        out.update(prefix_lengths=torch.tensor(lengths), suffix_input_ids=suffix_ids, suffix_attention_mask=suffix_mask)
    return out
