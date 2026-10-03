"""Self-contained inference exports: all weights once, configuration and tokenizer.

Training checkpoints remain compact overlays. A published model stores the full
state, including frozen embeddings and decision layers. Safetensors deduplicates
shared embeddings/output weights; loading constructs the architecture from local
configuration and strictly restores every tensor without fetching a base model.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path

import torch

from sors.core.decision import architecture_config, decision_config
from sors.core.model import adapter_config, prepare_model
from sors.core.tokens import CONTEXT_TOKENS, D_TOKENS, TYPE_TOKENS

FORMAT = "sors-pretrained-v1"
_DTYPES = {name: getattr(torch, name) for name in ("float32", "float16", "bfloat16", "float64")}


def _token_ids(tokenizer, ids, vocab_size):
    names = D_TOKENS + TYPE_TOKENS + CONTEXT_TOKENS
    if (not len(D_TOKENS) <= len(ids) <= len(names) or len(set(ids)) != len(ids)
            or any(type(i) is not int or not 0 <= i < vocab_size for i in ids)
            or tokenizer.convert_tokens_to_ids(names[:len(ids)]) != ids):
        raise ValueError("exported tokenizer and trainable token IDs must match the model embeddings")


def save_pretrained(model, tokenizer, training_config, directory) -> None:
    """Write a new complete model directory without changing the live model.

    Existing directories are refused. This format is for inference with the Sors
    runtime; it carries the same model structure as the training checkpoint and
    preserves the dtype of every parameter, including float32 decision layers.
    """
    from safetensors.torch import save_model

    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"export directory already exists: {directory}")
    embedding = model.get_input_embeddings()
    ids = embedding.ids.tolist()
    config = model.config.get_text_config().to_dict()
    config.pop("_name_or_path", None)
    _token_ids(tokenizer, ids, config["vocab_size"])
    training = asdict(training_config) if is_dataclass(training_config) else dict(training_config)
    architecture = architecture_config(model)
    if architecture["kind"] == "candidate":
        training["candidate_prefix_cache"] = model.candidate_prefix_cache
    dtypes = {name: str(value.dtype).removeprefix("torch.") for name, value in model.named_parameters()}
    if any(dtype not in _DTYPES for dtype in dtypes.values()):
        raise ValueError("complete exports require floating-point model parameters")
    record = {
        "model_type": "sors", "format": FORMAT, "text_config": config,
        "architecture": architecture, "adapter": adapter_config(model),
        "trainable_token_ids": ids, "training_config": training,
        "backbone_dtype": str(embedding.base.weight.dtype).removeprefix("torch."),
        "parameter_dtypes": dtypes,
    }
    directory.mkdir(parents=True)
    save_model(model, directory / "model.safetensors", metadata={"format": "pt", "sors_format": FORMAT})
    tokenizer.save_pretrained(directory)
    with (directory / "config.json").open("x") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def load_pretrained(directory, *, device="cuda", attn_implementation="sdpa", allow_kernel_download=False):
    """Return (model, tokenizer, decision IDs, training config) using local artifacts.

    Saved mixed precision is preserved. Optional attention kernels use the same
    selection policy as checkpoint loading; model/tokenizer assets stay local.
    """
    from safetensors.torch import load_model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from sors.core.attention import _configure_linear_attention, resolve_attention

    directory = Path(directory)
    record = json.loads((directory / "config.json").read_text())
    if record.get("format") != FORMAT:
        raise ValueError(f"unsupported complete-model format: {record.get('format')!r}")
    text = dict(record["text_config"])
    config = AutoConfig.for_model(text.pop("model_type"), **text)
    decision = decision_config(record["architecture"])
    dtype = _DTYPES[record["backbone_dtype"]]
    tok = AutoTokenizer.from_pretrained(directory, config=config, local_files_only=True)
    ids = record["trainable_token_ids"]
    _token_ids(tok, ids, config.vocab_size)
    device = torch.device(device)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
        choice = resolve_attention(attn_implementation, device=device, dtype=dtype,
                                   allow_kernel_download=allow_kernel_download, head_dim=head_dim,
                                   attention_dropout=getattr(config, "attention_dropout", 0.0))
        raw = AutoModelForCausalLM.from_config(config, dtype=dtype,
                                              attn_implementation=choice.implementation).to(device)
        model = prepare_model(raw, ids, lora_dropout=0.0, **record["adapter"], decision=decision)
        parameters = dict(model.named_parameters())
        if set(parameters) != set(record["parameter_dtypes"]):
            raise ValueError("exported parameter dtype manifest differs from the configured model")
        # Cast parameters individually: converting the whole model would also
        # round non-persistent rotary buffers that the backbone keeps in fp32.
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.data = parameter.data.to(_DTYPES[record["parameter_dtypes"][name]])
        load_model(model, directory / "model.safetensors", strict=True, device=str(device))
        _configure_linear_attention(raw, allow_kernel_download=allow_kernel_download)
    training = record["training_config"]
    if record["architecture"]["kind"] == "candidate":
        model.set_candidate_prefix_cache(training.get("candidate_prefix_cache", "off"))
    return model.eval(), tok, ids[:len(D_TOKENS)], training
