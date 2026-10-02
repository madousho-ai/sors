"""Content-based decision layers. Candidate rows share every parameter."""

from __future__ import annotations

import math
import threading
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from functools import partial

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

_OPTION_CHUNK_OWNER = ContextVar('sors_option_chunk_owner', default=None)
# Bound the activation scale of a whole option-chunk replay. Larger chunks keep
# native layer checkpoints; the long common text always keeps them.
_OPTION_CHECKPOINT_ELEMENTS = 128 * 1024 * 1024


def _checkpoint_layer(owner, function, *args, **kwargs):
    if _OPTION_CHUNK_OWNER.get() is owner:
        return function(*args, **kwargs)
    return checkpoint(function, *args, use_reentrant=False, **kwargs)


@dataclass(frozen=True)
class DecisionConfig:
    kind: str = "minimal"
    dim: int = 128
    heads: int = 4
    blocks: int = 2
    layers: tuple[int, ...] | None = None
    feedback: bool = False
    option_batch_size: int = 32

    def __post_init__(self):
        if self.kind not in ("minimal", "structural", "candidate"):
            raise ValueError(f"unknown decision architecture {self.kind!r}")
        if self.dim < 2 or min(self.heads, self.option_batch_size) < 1 or self.dim % self.heads:
            raise ValueError("decision dim must be >= 2 and divisible by positive heads; batch count must be positive")
        if self.blocks < (0 if self.kind == "candidate" else 1):
            raise ValueError("candidate set-block count must be >= 0; other decision architectures need >= 1")
        if self.layers is not None:
            object.__setattr__(self, "layers", tuple(self.layers))
        if self.kind == "candidate" and (self.feedback or self.layers):
            raise ValueError("candidate uses post-encoder SetBlocks; feedback and backbone layer indices are unsupported")

    def resolve(self, depth: int):
        if self.kind == "candidate":
            return replace(self, layers=())
        if self.blocks > depth:
            raise ValueError(f"{self.blocks} decision blocks need at least that many backbone layers; got {depth}")
        layers = self.layers
        if layers is None:
            layers = (tuple(range(depth - self.blocks, depth)) if self.kind == "minimal" else
                      tuple((i + 1) * (depth - 1) // self.blocks for i in range(self.blocks)))
        if len(layers) != self.blocks or tuple(sorted(set(layers))) != layers or not all(0 <= i < depth for i in layers):
            raise ValueError("decision layers must be increasing distinct backbone indices, one per block")
        return replace(self, layers=layers)


def architecture_config(model) -> dict:
    cfg = getattr(model, "decision_config", None)
    if cfg is None:
        return {"kind": "slots"}
    out = asdict(cfg)
    out["layers"] = list(cfg.layers) if cfg.layers is not None else None
    return out


def decision_config(record: dict | None) -> DecisionConfig | None:
    if record is None or record == {"kind": "slots"}:
        return None
    return DecisionConfig(**record)


class DecisionBlock(nn.Module):
    """Read text evidence, compare a set of candidates, optionally write back.

    No candidate-position embeddings. Padding is excluded from both attention
    directions. The feedback output starts at zero, preserving the text stream
    at initialization while allowing that output projection to learn immediately.
    """

    def __init__(self, dim: int, hidden: int, heads: int, *, feedback: bool = False):
        super().__init__()
        self.memory_norm = nn.LayerNorm(hidden)
        self.memory = nn.Linear(hidden, dim)
        self.read_norm = nn.LayerNorm(dim)
        self.read = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.set_norm = nn.LayerNorm(dim)
        self.compare = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.write = nn.MultiheadAttention(dim, heads, batch_first=True) if feedback else None
        self.write_out = nn.Linear(dim, hidden, bias=False) if feedback else None
        if self.write_out is not None:
            nn.init.zeros_(self.write_out.weight)

    def forward(self, u, h, option_mask, text_mask):
        memory = self.memory(self.memory_norm(h))
        read, _ = self.read(self.read_norm(u), memory, memory, key_padding_mask=~text_mask, need_weights=False)
        u = u + read
        norm = self.set_norm(u)
        compared, _ = self.compare(norm, norm, norm, key_padding_mask=~option_mask, need_weights=False)
        u = u + compared
        u = (u + self.ff(self.ff_norm(u))).masked_fill(~option_mask[..., None], 0)
        if self.write is not None:
            write, _ = self.write(memory, u, u, key_padding_mask=~option_mask, need_weights=False)
            h = h + self.write_out(write).masked_fill(~text_mask[..., None], 0)
        return u, h


class CandidateSetBlock(nn.Module):
    """Set interaction after joint encoding; preserve the native hidden width.

    Attention uses a configurable bottleneck. Both residual paths return to the
    backbone's hidden dimension, so the zero-block baseline reads its original
    pretrained feature directly with the same RMSNorm/scalar head.
    """

    def __init__(self, hidden: int, dim: int, heads: int):
        super().__init__()
        self.norm = nn.RMSNorm(hidden, eps=1e-6)
        self.project = nn.Linear(hidden, dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.output = nn.Linear(dim, hidden, bias=False)
        self.ff_norm = nn.RMSNorm(hidden, eps=1e-6)
        self.ff = nn.Sequential(nn.Linear(hidden, 4 * dim), nn.GELU(), nn.Linear(4 * dim, hidden))

    def forward(self, u, valid):
        projected = self.project(self.norm(u))
        mixed, _ = self.attention(projected, projected, projected, key_padding_mask=~valid, need_weights=False)
        u = u + self.output(mixed)
        return (u + self.ff(self.ff_norm(u))).masked_fill(~valid[..., None], 0)


class DecisionModel(nn.Module):
    """A Qwen text backbone with a set-valued side stream.

    Each invocation owns its side-stream state and removes temporary taps in
    finally. Read-only decisions run after the text backbone, allowing
    native layer-local checkpoints and independent decision-block checkpoints.
    Coupled paths rebuild their whole graph under a lock during recomputation.
    """

    def __init__(self, base, config: DecisionConfig, adapter: dict, grad_ckpt: bool = False):
        super().__init__()
        self.base = base
        raw = base.get_base_model() if hasattr(base, "get_base_model") else base
        if raw.config.model_type not in ("qwen3", "qwen3_5_text"):
            raise ValueError("decision architectures currently support Qwen3 and Qwen3.5 text backbones")
        self.decision_config = config.resolve(len(raw.model.layers))
        self.adapter = dict(adapter)
        self.config = raw.config
        self.checkpoint_forward = grad_ckpt
        self.candidate_prefix_cache = "auto" if config.kind == "candidate" else "off"
        self._read_only = config.kind in ("minimal", "structural") and not config.feedback
        self._checkpoint_owner = object()
        if self._read_only and grad_ckpt:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            base.enable_input_require_grads()
            if config.kind == "structural":
                for layer in raw.model.layers:
                    layer._gradient_checkpointing_func = partial(_checkpoint_layer, self._checkpoint_owner)
        else:
            base.gradient_checkpointing_disable()
        hidden = self.config.hidden_size
        if config.kind == "candidate":
            self.blocks = nn.ModuleList([CandidateSetBlock(hidden, config.dim, config.heads)
                                         for _ in range(config.blocks)])
            self.option_norm = nn.RMSNorm(hidden, eps=1e-6)
            self.scorer = nn.Linear(hidden, 1, bias=False)
            decision_layers = (self.blocks, self.option_norm, self.scorer)
        else:
            self.option_proj = nn.Linear(hidden, config.dim)
            self.blocks = nn.ModuleList([DecisionBlock(config.dim, hidden, config.heads, feedback=config.feedback)
                                         for _ in range(config.blocks)])
            self.query = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, config.dim))
            self.option_norm = nn.LayerNorm(config.dim)
            decision_layers = (self.option_proj, self.blocks, self.query, self.option_norm)
        # Keep new decision parameters in fp32, just like PEFT adapters; the
        # training loop's master weights still handle a low-precision backbone.
        device = next(base.parameters()).device
        for layer in decision_layers:
            layer.to(device=device)
        self._forward_lock = threading.RLock()

    @property
    def decoder(self):
        raw = self.base.get_base_model() if hasattr(self.base, "get_base_model") else self.base
        return raw.model

    def get_input_embeddings(self):
        return self.base.get_input_embeddings()

    def _encode(self, ids, mask):
        # Explicit per-sequence positions preserve padding and branch independence
        # for both rotary attention and the hybrid text backbone.
        positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
        return self.decoder(input_ids=ids, attention_mask=mask, position_ids=positions,
                            use_cache=False).last_hidden_state

    def _encode_option_chunk(self, ids, mask, dtype):
        scale = ids.numel() * self.config.hidden_size * len(self.decoder.layers)
        if self._read_only and self.decision_config.kind == "structural" and scale <= _OPTION_CHECKPOINT_ELEMENTS:
            # The enclosing chunk checkpoint already discards its activations.
            # A call-local owner avoids a third forward without changing another
            # thread/model's native checkpoint policy during backward replay.
            token = _OPTION_CHUNK_OWNER.set(self._checkpoint_owner)
            try:
                return self._encode(ids, mask)[:, -1].to(dtype)
            finally:
                _OPTION_CHUNK_OWNER.reset(token)
        return self._encode(ids, mask)[:, -1].to(dtype)

    def _encode_options(self, batch, valid, dtype, option_batch_size):
        ids, masks = batch["option_input_ids"][valid], batch["option_attention_mask"][valid]
        chunk = min(option_batch_size or self.decision_config.option_batch_size, self.decision_config.option_batch_size)
        lengths = masks.sum(1)
        padded = torch.nn.functional.pad(lengths, (0, (-len(lengths)) % chunk))
        widths = padded.reshape(-1, chunk).amax(1).cpu().tolist()
        pieces = []
        for start, width in zip(range(0, len(ids), chunk), widths):
            width = int(width)  # Float 0/1 masks produce floats in the CPU list.
            part_ids, part_mask = ids[start:start + chunk], masks[start:start + chunk]
            args = (part_ids[:, -width:], part_mask[:, -width:], dtype)
            if self._read_only and self.checkpoint_forward and self.training and torch.is_grad_enabled():
                # Keep one readout per option between chunks. Large chunks also
                # retain native layer checkpoints to bound backward memory.
                encoded = checkpoint(self._encode_option_chunk, *args, use_reentrant=False)
            else:
                encoded = self._encode_option_chunk(*args)
            pieces.append(encoded)
        return torch.cat(pieces)

    def _output_logits(self, scores, slots, valid):
        # The vocabulary tensor is only the external coordinate protocol.
        logits = scores.new_full((len(slots), self.config.vocab_size), float("-inf"))
        rows = torch.arange(len(slots), device=slots.device)[:, None].expand_as(slots)
        logits[rows[valid], slots[valid]] = scores[valid]
        return logits.float()

    def set_candidate_prefix_cache(self, mode: str):
        if mode not in ("auto", "on", "off"):
            raise ValueError("candidate prefix cache mode must be auto, on or off")
        if self.decision_config.kind != "candidate" and mode != "off":
            raise ValueError("prefix sharing is only available for the candidate architecture")
        self.candidate_prefix_cache = mode

    def _encoder_frozen_for_batch(self, batch):
        if not torch.is_grad_enabled():
            return True
        embedding = self.get_input_embeddings()
        rows = embedding.rows
        if any(p.requires_grad for p in self.decoder.parameters() if p is not rows):
            return False
        if rows.requires_grad:
            # Newly installed type/context rows can remain trainable even when
            # the original backbone is frozen. Check the actual encoded input.
            ids, mask = batch["option_input_ids"], batch["option_attention_mask"].bool()
            if bool(((embedding.lut[ids] >= 0) & mask).any()):
                return False
        return True

    def uses_shared_prefix(self, batch):
        if self.decision_config.kind != "candidate" or self.candidate_prefix_cache == "off":
            return False
        frozen = self._encoder_frozen_for_batch(batch)
        if not frozen and self.candidate_prefix_cache == "on":
            raise ValueError("shared prefix cannot discard trainable encoder/token gradients; use auto or off")
        if "prefix_lengths" not in batch:
            if self.candidate_prefix_cache == "on":
                raise ValueError("shared prefix requires a candidate batch with token boundaries")
            return False
        return frozen

    def _candidate_output(self, encoded, slots):
        valid = slots >= 0
        u = encoded.new_zeros((*slots.shape, encoded.shape[-1]))
        u[valid] = encoded
        for block in self.blocks:
            u = block(u, valid)
        scores = self.scorer(self.option_norm(u)).squeeze(-1)  # T=1, shared scalar head.
        return self._output_logits(scores, slots, valid)

    def _read_only_block(self, index, u, h, positions, valid, mask):
        """Explicit inputs make each decision block independently recomputable."""
        dtype = self.option_proj.weight.dtype
        if u is None:
            chosen = h.gather(1, positions[..., None].expand(-1, -1, h.shape[-1]))
            u = self.option_proj(chosen.to(dtype))
        return self.blocks[index](u, h.to(dtype), valid, mask)[0]

    def _forward_read_only(self, batch, option_batch_size=None):
        with self._forward_lock:
            slots = batch["slot_ids"]
            valid = slots >= 0
            mask = batch["attention_mask"].bool()
            if not valid.any(1).all() or not mask.any(1).all():
                raise ValueError("decision batches need at least one option and one text token per example")
            u = None
            positions = None
            if self.decision_config.kind == "structural":
                dtype = self.option_proj.weight.dtype
                encoded = self.option_proj(self._encode_options(batch, valid, dtype, option_batch_size))
                u = encoded.new_zeros((*slots.shape, encoded.shape[-1]))
                u[valid] = encoded
            else:
                positions = batch["option_positions"].clamp_min(0)
            states = {}
            owner = threading.get_ident()

            def capture(index):
                def record(_layer, _args, h):
                    # Another thread can be replaying native checkpoints while
                    # this forward owns the temporary capture hooks.
                    if threading.get_ident() != owner:
                        return
                    if not isinstance(h, torch.Tensor):
                        raise TypeError("decision taps require a tensor-returning text decoder layer")
                    # Keep the raw layer output and its gradient edge; the final
                    # decoder output has an additional norm used only by query.
                    states[index] = h
                return record

            handles = []
            try:
                for j, i in enumerate(self.decision_config.layers):
                    handles.append(self.decoder.layers[i].register_forward_hook(capture(j)))
                h = self._encode(batch["input_ids"], batch["attention_mask"])
            finally:
                for handle in handles:
                    handle.remove()
            for j in range(len(self.blocks)):
                args = (j, u, states[j], positions, valid, mask)
                if self.checkpoint_forward and self.training and torch.is_grad_enabled():
                    u = checkpoint(self._read_only_block, *args, use_reentrant=False)
                else:
                    u = self._read_only_block(*args)
            dtype = self.option_proj.weight.dtype
            scores = (self.option_norm(u) * self.query(h[:, -1].to(dtype))[:, None]).sum(-1) / math.sqrt(u.shape[-1])
            return self._output_logits(scores, slots, valid)

    def _forward_batch(self, batch, option_batch_size=None):
        with self._forward_lock:
            slots = batch["slot_ids"]
            valid = slots >= 0
            mask = batch["attention_mask"].bool()
            if not valid.any(1).all() or not mask.any(1).all():
                raise ValueError("decision batches need at least one option and one text token per example")
            if self.decision_config.kind == "candidate":
                encoded = self._encode_options(batch, valid, self.option_norm.weight.dtype, option_batch_size)
                return self._candidate_output(encoded, slots)
            dtype = self.option_proj.weight.dtype
            u = None
            if self.decision_config.kind == "structural":
                encoded = self.option_proj(self._encode_options(batch, valid, dtype, option_batch_size))
                u = encoded.new_zeros((*slots.shape, encoded.shape[-1]))
                u[valid] = encoded

            def tap(index):
                def update(_layer, _args, h):
                    nonlocal u
                    if not isinstance(h, torch.Tensor):
                        raise TypeError("decision taps require a tensor-returning text decoder layer")
                    if u is None:
                        positions = batch["option_positions"].clamp_min(0)
                        chosen = h.gather(1, positions[..., None].expand(-1, -1, h.shape[-1]))
                        u = self.option_proj(chosen.to(dtype))
                    u, updated = self.blocks[index](u, h.to(dtype), valid, mask)
                    return updated.to(h.dtype) if self.decision_config.feedback else h
                return update

            handles = []
            try:
                for j, i in enumerate(self.decision_config.layers):
                    handles.append(self.decoder.layers[i].register_forward_hook(tap(j)))
                h = self._encode(batch["input_ids"], batch["attention_mask"])
            finally:
                for handle in handles:
                    handle.remove()
            scores = (self.option_norm(u) * self.query(h[:, -1].to(dtype))[:, None]).sum(-1) / math.sqrt(u.shape[-1])
            # D tokens are the external coordinate protocol. This model defines
            # a K-way distribution; it never invokes or trains the LM vocabulary head.
            return self._output_logits(scores, slots, valid)

    def forward_batch(self, batch, *, option_batch_size=None):
        if self._read_only:
            return self._forward_read_only(batch, option_batch_size)
        if self.decision_config.kind == "candidate":
            shared = self.uses_shared_prefix(batch)
            if self._encoder_frozen_for_batch(batch):
                from sors.core.candidate_cache import frozen_encoder, shared_candidate_hidden
                valid = batch["slot_ids"] >= 0
                if not valid.any(1).all():
                    raise ValueError("candidate batch needs at least one option per example")
                with self._forward_lock, frozen_encoder(self.decoder):
                    if shared:
                        chunk = min(option_batch_size or self.decision_config.option_batch_size, self.decision_config.option_batch_size)
                        encoded = shared_candidate_hidden(self.decoder, batch, valid, chunk).to(self.option_norm.weight.dtype)
                    else:
                        encoded = self._encode_options(batch, valid, self.option_norm.weight.dtype, option_batch_size)
                # Frozen feature extraction lives outside the head checkpoint;
                # backward must not prefill the expensive context again.
                if self.checkpoint_forward and self.training and torch.is_grad_enabled():
                    return checkpoint(self._candidate_output, encoded, batch["slot_ids"], use_reentrant=False)
                return self._candidate_output(encoded, batch["slot_ids"])
        if self.checkpoint_forward and self.training and torch.is_grad_enabled():
            return checkpoint(self._forward_batch, batch, option_batch_size, use_reentrant=False)
        return self._forward_batch(batch, option_batch_size)
