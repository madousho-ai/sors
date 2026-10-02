"""CUDA equivalence of shared prefixes, optionally with cached real Qwen weights.

No training run, model download or dependency installation. The real-model probe
uses local_files_only and reports hidden-state and probability error separately.
"""

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import json
import importlib
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from _runner import run
from test_candidate import candidate_batch, candidate_model, examples
from test_candidate_prefix import trace
from sors.core.batch import collate
from sors.core.candidate_cache import frozen_encoder, shared_candidate_hidden
from sors.core.decision import DecisionConfig
from sors.core.model import prepare_model
from sors.core.tokens import install_context_tokens, install_d_tokens, install_type_tokens


def compare(m, data, label, *, probability_atol=5e-3):
    m.eval()
    with torch.no_grad():
        m.set_candidate_prefix_cache("off")
        with trace(m) as complete_calls:
            full = m.forward_batch(data)
        m.set_candidate_prefix_cache("on")
        with trace(m) as cached_calls:
            cached = m.forward_batch(data)
        valid = data["slot_ids"] >= 0
        with frozen_encoder(m.decoder):
            full_h = m._encode_options(data, valid, torch.float32, None)
            cache_h = shared_candidate_hidden(m.decoder, data, valid, m.decision_config.option_batch_size).float()
        hidden_relative = float((full_h - cache_h).norm() / full_h.norm().clamp_min(1e-12))
        full_p, cache_p = full.softmax(-1), cached.softmax(-1)
        probability_error = float((full_p - cache_p).abs().max())
        logit_error = float((full[torch.isfinite(full)] - cached[torch.isfinite(cached)]).abs().max())
        prefills = sum(c["cache"] and c["past"] == 0 for c in cached_calls)
        tokens_full = sum(c["ids"].numel() for c in complete_calls)
        tokens_shared = sum(c["ids"].numel() for c in cached_calls)
        report = {"label": label, "prefix_lengths": data["prefix_lengths"].tolist(),
                  "hidden_max_abs": float((full_h - cache_h).abs().max()), "hidden_relative_l2": hidden_relative,
                  "logit_max_abs": logit_error, "probability_max_abs": probability_error,
                  "same_argmax": bool((full.argmax(-1) == cached.argmax(-1)).all()),
                  "top_two_margin": (full_p.topk(2, dim=-1).values[:, 0] - full_p.topk(2, dim=-1).values[:, 1]).tolist(),
                  "prefills": prefills, "processed_full": tokens_full, "processed_shared": tokens_shared}
        print(json.dumps(report), flush=True)
        assert torch.isfinite(full_h).all() and torch.isfinite(cache_h).all()
        assert hidden_relative < 0.02 and probability_error < probability_atol, report
        # BF16 changed the winner in a tiny random model with a 0.00145 margin;
        # its fp32 full/cached distributions agreed within 1e-7. Validate the
        # probability tolerance and require rank stability outside that margin.
        separated = torch.tensor(report["top_two_margin"], device=full.device) > 2 * probability_atol
        assert bool(((full.argmax(-1) == cached.argmax(-1)) | ~separated).all()), report
        assert prefills == 1 and tokens_shared < tokens_full, report
        torch.testing.assert_close(m.forward_batch(data), cached, atol=0, rtol=0)


def tiny(family, blocks):
    print(f"START CUDA prefix {family} blocks={blocks}", flush=True)
    m, tok, d, _ = candidate_model(blocks, family)
    m.base.to(device="cuda", dtype=torch.bfloat16)
    m.to(device="cuda")
    ex = replace(examples()[0], query="long context " * 32,
                 option_names=["red", "long blue", "long long green"])
    data = {k: v.cuda() for k, v in candidate_batch(tok, d, [ex]).items()}
    compare(m, data, f"tiny-{family}-{blocks}")
    m.train()
    m.checkpoint_forward = True
    m.zero_grad(set_to_none=True)
    with trace(m) as calls:
        loss = -m.forward_batch(data).log_softmax(-1)[:, d[0]].mean()
        loss.backward()
    assert sum(c["cache"] and c["past"] == 0 for c in calls) == 1
    assert m.scorer.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert all(p.grad is None for p in m.base.parameters())


def real_model(args):
    from transformers import AutoTokenizer
    from sors.core.attention import load_causal_lm
    print(f"START real prefix {args.real_model} revision={args.revision}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.real_model, revision=args.revision, local_files_only=True)
    d = install_d_tokens(tok)
    ids = d + install_type_tokens(tok) + install_context_tokens(tok)
    lm, attention = load_causal_lm(args.real_model, revision=args.revision, local_files_only=True,
                                    attn_implementation=args.attn_implementation,
                                    allow_kernel_download=args.allow_kernel_download)
    print(json.dumps({"attention": asdict(attention), "model_type": lm.config.model_type}), flush=True)
    torch.manual_seed(0)
    m = prepare_model(lm, ids, None, None, 0., trainable="decision-only",
                      decision=DecisionConfig(kind="candidate", blocks=2, option_batch_size=2))
    for repeats in (4, 128, 600):
        print(f"START real context repeats={repeats}", flush=True)
        ex = replace(examples()[0],
                     query=("The audit records show that the request was reviewed by the team. " * repeats)
                           + "The final decision was approval.",
                     question="Which conclusion is supported by the records?",
                     options=[0, 1, 2, 3], label=0, gold_idx=0,
                     option_names=["The request was approved.", "The request was rejected.",
                                   "The team is still waiting for additional evidence.", "The records contain no decision."])
        data = {k: v.to("cuda") for k, v in collate([ex], tok, d, 4, max_length=8192, architecture="candidate").items()}
        compare(m, data, f"real-{repeats}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-model")
    parser.add_argument("--revision")
    parser.add_argument("--attn-implementation", default="auto")
    parser.add_argument("--allow-kernel-download", action="store_true")
    parser.add_argument("--fa2-kernel-path", help="load an already built local FA2 kernel project")
    parser.add_argument("--conv-kernel-path", help="load an already built local causal-conv1d kernel project")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; run on the GPU server")
    torch.cuda.set_per_process_memory_fraction(0.2)
    if args.real_model:
        paths = {repo: Path(path) for repo, path in (
            ("kernels-community/flash-attn2", args.fa2_kernel_path),
            ("kernels-community/causal-conv1d", args.conv_kernel_path)) if path}
        if paths:
            import kernels
            from kernels import _versions
            modules = {repo: kernels.get_local_kernel(path) for repo, path in paths.items()}
            original_resolve, original_get = _versions.resolve_version_spec_as_ref, kernels.get_kernel
            def resolve(repo_id, version):
                return SimpleNamespace(target_commit=paths[repo_id].name) if repo_id in paths else original_resolve(repo_id, version)
            def get(repo_id, *a, **kw):
                return modules[repo_id] if repo_id in modules else original_get(repo_id, *a, **kw)
            print(json.dumps({"local_kernels": {repo: module.__file__ for repo, module in modules.items()}}), flush=True)
            # Override selection only in this test process. The actual compiled
            # functions run unchanged, without Hub version/snapshot downloads.
            with ExitStack() as stack:
                stack.enter_context(patch.object(_versions, "resolve_version_spec_as_ref", side_effect=resolve))
                for name, attribute in (("kernels", "get_kernel"), ("kernels.layer.func", "get_kernel"),
                                        ("kernels.layer.layer", "get_kernel"),
                                        ("transformers.integrations.hub_kernels", "get_kernel_hub")):
                    stack.enter_context(patch.object(importlib.import_module(name), attribute, side_effect=get))
                real_model(args)
        else:
            real_model(args)
    else:
        run({f"test_prefix_cuda_{family}_{blocks}":
             (lambda family=family, blocks=blocks: tiny(family, blocks))
             for family in ("qwen3", "qwen35") for blocks in (0, 2)})
