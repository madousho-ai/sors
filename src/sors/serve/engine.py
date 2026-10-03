"""推理引擎: 一份训好的存档, 回答一个 state 上的一组 API 问题, 给出每道题在它菜单各行上的分布.

一个请求怎么跑:
  1. 每道题照 serve.menus 排成菜单, 按训练时的 context-first 模板切成两段: state 那段所有题共用, 问题那段各题自己的
  2. state 那段前向一次, 得到它的 KV cache (core.cache.prefix_cache)
  3. 问题按长度排好分组, 每组的分支接在 cache 后面一次前向 (core.cache.branch_logits).
     一组的 cache 是 组员数 × (state + 组内最长的问题) 个 token, 不超过 max_batch_tokens; 一题超了就单独一组
  4. 每道题在自己菜单的 k 个 D 码上做 softmax. 与评估 (evaluation.scoring) 读的是同一个分布
state 加上最长的那道题超过 max_tokens 时整个请求拒收, 不碰模型.
prompts 给出第 1 步的两段文本 (模型读到的提示原文), 不跑模型; evaluate 用的就是这两段.
GPU 同一时刻只跑一个请求 (一把锁), 并发的请求排队.

load_engine 从基模 + 存档搭引擎: LoRA 照档里记的形状挂上再合并进权重 (部署形态, 比旁路挂法快),
档里记的 type_marker、context_marker 照搬 (没记的是加这一项之前的档, 不包). menu-first 训的档没有可共享的 state 前缀, 拒收.
调用方的 state、问句、选项名照 core.prompt.encode_prompts 编码: 里面写着的 <|D5|> 之类是普通文字.
"""

from __future__ import annotations

import json
import pathlib
import threading
from dataclasses import dataclass

import torch

from sors.core.cache import branch_logits, prefix_cache
from sors.core.checkpoint import prepare_from_checkpoint
from sors.core.prompt import encode_prompts, prompt_pieces, split_prompt
from sors.core.tokens import install_context_tokens, install_d_tokens, install_type_tokens
from sors.serve.menus import to_example

LAYOUT = "context-first"  # state 在前才有可共享的前缀


class RequestTooLong(ValueError):
    """state 加最长的问题超过引擎的 max_tokens."""


@dataclass(frozen=True)
class Evaluation:
    probs: dict[str, list[float]]  # 问题 id -> 菜单各行的概率, 行序见 serve.menus
    input_tokens: int  # state 一次 + 每道题各自那段


class Engine:
    def __init__(self, lm, tok, d_ids: list[int], context_label: str = "", type_marker: bool = False,
                 context_marker: bool = False, max_tokens: int = 8192, max_batch_tokens: int = 16384,
                 candidate_prefix_cache: str | None = None):
        self.lm = lm.eval()
        if candidate_prefix_cache is not None:
            if getattr(getattr(lm, "decision_config", None), "kind", None) != "candidate":
                raise ValueError("candidate prefix cache override requires a candidate model")
            lm.set_candidate_prefix_cache(candidate_prefix_cache)
        self.tok = tok
        self.d_ids = d_ids
        self.context_label = context_label
        self.type_marker = type_marker
        self.context_marker = context_marker
        self.max_tokens = max_tokens
        self.max_batch_tokens = max_batch_tokens
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        self._lock = threading.Lock()

    def prompts(self, state, questions: dict) -> dict[str, tuple[str, str]]:
        """{问题 id: (state 段, 问题段)}: 模型读到的提示原文, 两段拼起来就是训练模板下的整条提示.
        state 段所有题相同 (只前向一次), 问题段各题自己的, 以 'Answer:' 收尾. 不碰模型.
        标签默认为空, state 段就是 state 本身; 给了 context_label 才在前面加「<标签>: 」."""
        kind = getattr(getattr(self.lm, "decision_config", None), "kind", None)
        if kind == "candidate":
            from sors.core.decision_batch import candidate_pieces
            out = {}
            for qid, q in questions.items():
                prefix, suffixes = candidate_pieces(to_example(q, state, self.context_label), self.type_marker,
                                                    self.context_marker)
                out[qid] = ("".join(prefix), "\n\n".join(f"Candidate branch {i + 1} (prefix + this suffix):\n{''.join(p)}"
                                                        for i, p in enumerate(suffixes)))
            return out
        if kind == "structural":
            from sors.core.decision_batch import structural_pieces
            out = {}
            for qid, q in questions.items():
                memory, options = structural_pieces(to_example(q, state, self.context_label), self.type_marker,
                                                     self.context_marker)
                out[qid] = ("".join(memory), "\n\n".join(f"Independent option branch {i + 1}:\n{''.join(p)}"
                                                       for i, p in enumerate(options)))
            return out
        return {qid: split_prompt(to_example(q, state, self.context_label), LAYOUT, self.type_marker, self.context_marker)
                for qid, q in questions.items()}

    def evaluate(self, state, questions: dict) -> Evaluation:
        """questions: {问题 id: serve.api 的 Noul / Choice / Score}. 返回的 probs 与 questions 同序."""
        exs = {qid: to_example(q, state, self.context_label) for qid, q in questions.items()}
        if getattr(self.lm, "decision_config", None) is not None:
            return self._evaluate_decisions(exs)
        pieces = {qid: prompt_pieces(ex, LAYOUT, self.type_marker, self.context_marker) for qid, ex in exs.items()}
        ctx, *branches = encode_prompts(self.tok, [next(iter(pieces.values()))[0]] + [q for _, q in pieces.values()])
        segs = dict(zip(pieces, branches))
        longest = max(len(s) for s in segs.values())
        if len(ctx) + longest > self.max_tokens:
            raise RequestTooLong(f"the state ({len(ctx)} tokens) plus the longest question ({longest} tokens) "
                                 f"come to {len(ctx) + longest} tokens; this server takes at most {self.max_tokens}")
        probs = {}
        with self._lock:
            cache = prefix_cache(self.lm, ctx)
            for group in self._groups(segs, len(ctx)):
                logits = branch_logits(self.lm, cache, [segs[qid] for qid in group], self.pad_id)
                for row, qid in enumerate(group):
                    slots = [self.d_ids[c] for c in exs[qid].slot_codes]
                    probs[qid] = torch.softmax(logits[row, slots], dim=-1).tolist()
        return Evaluation({qid: probs[qid] for qid in questions}, len(ctx) + sum(len(s) for s in segs.values()))

    def _evaluate_decisions(self, exs):
        """Each request is preflighted in full. New architectures use no KV cache.

        Structural prompt previews show the memory and independent option branches;
        input_tokens counts all encoded streams, including their repeated questions.
        """
        from sors.core.decision_batch import collate_decisions
        batches, tokens = {}, 0
        for qid, ex in exs.items():
            try:
                b = collate_decisions([ex], self.tok, self.d_ids, len(ex.options), LAYOUT, self.max_tokens,
                                      self.type_marker, self.context_marker, self.lm.decision_config.kind,
                                      truncate=False)
            except ValueError as exc:
                raise RequestTooLong(str(exc)) from exc
            batches[qid] = b
            if self.lm.decision_config.kind != "candidate":
                tokens += int(b["attention_mask"].sum())
                if "option_attention_mask" in b:
                    tokens += int(b["option_attention_mask"].sum())
        probs = {}
        device = next(self.lm.parameters()).device
        with self._lock, torch.inference_mode():
            for qid, b in batches.items():
                b = {k: v.to(device) for k, v in b.items()}
                shared = False
                if self.lm.decision_config.kind == "candidate":
                    from sors.core.candidate_cache import shared_input_tokens
                    shared = self.lm.uses_shared_prefix(b)
                    tokens += (shared_input_tokens(b) if shared else
                               int(b["option_attention_mask"].sum()))
                option_batch = None
                if "option_attention_mask" in b:
                    width = int(b["option_attention_mask"].sum(-1).max())
                    option_batch = max(1, self.max_batch_tokens // width)
                    if shared:
                        from sors.core.candidate_cache import shared_branch_limit
                        option_batch = shared_branch_limit(b, self.max_batch_tokens)
                logits = self.lm.forward_batch(b, option_batch_size=option_batch)
                probs[qid] = logits[0, b["slot_ids"][0]].softmax(-1).tolist()
        return Evaluation(probs, tokens)

    def _groups(self, segs: dict[str, list[int]], n_ctx: int) -> list[list[str]]:
        """按问题长度从短到长装组, 组员数 × (state + 组内最长) 不超过 max_batch_tokens; 一题超了就自己一组."""
        groups: list[list[str]] = []
        for qid in sorted(segs, key=lambda q: len(segs[q])):
            g = groups[-1] if groups else None
            if g and (len(g) + 1) * (n_ctx + len(segs[qid])) <= self.max_batch_tokens:
                g.append(qid)
            else:
                groups.append([qid])
        return groups


def recorded_base_model(checkpoint) -> str | None:
    """存档所在 run 记的基模 (runs/<run>/result.json 的 args.model). 最终档在 run 目录里, 途中的档在它的 checkpoints/ 下.
    存档本身不记基模; 找不到就是 None."""
    p = pathlib.Path(checkpoint).resolve()
    for d in (p.parent, p.parent.parent):
        f = d / "result.json"
        if f.is_file():
            return json.loads(f.read_text()).get("args", {}).get("model")
    return None


def load_engine(checkpoint, base_model=None, context_label: str = "", device: str = "cuda",
                dtype: torch.dtype = torch.bfloat16, max_tokens: int = 8192, max_batch_tokens: int = 16384,
                candidate_prefix_cache: str | None = None, *, attn_implementation: str = "sdpa",
                allow_kernel_download: bool = False, local_files_only: bool = False) -> Engine:
    """Load a complete model directory or a legacy checkpoint plus its base.

    Complete exports preserve their saved mixed precision and use local model
    assets. The dtype argument controls legacy base-model loading.
    """
    from transformers import AutoTokenizer
    from sors.core.attention import load_causal_lm

    if pathlib.Path(checkpoint).is_dir():
        if base_model is not None:
            raise ValueError("a complete model directory already contains its base weights; omit base_model")
        from sors.core.pretrained import load_pretrained
        m, tok, d_ids, cfg = load_pretrained(checkpoint, device=device, attn_implementation=attn_implementation,
                                            allow_kernel_download=allow_kernel_download)
    else:
        if base_model is None:
            raise ValueError("a training checkpoint requires its base_model")
        tok = AutoTokenizer.from_pretrained(base_model, local_files_only=local_files_only)
        d_ids = install_d_tokens(tok)
        train_ids = d_ids + install_type_tokens(tok) + install_context_tokens(tok)
        lm, _ = load_causal_lm(base_model, device=device, dtype=dtype, attn_implementation=attn_implementation,
                               allow_kernel_download=allow_kernel_download, local_files_only=local_files_only)
        m, cfg = prepare_from_checkpoint(lm, train_ids, checkpoint)
    layout = cfg.get("layout", LAYOUT)
    if layout != LAYOUT:
        raise ValueError(f"{checkpoint} was trained {layout}; the server needs a {LAYOUT} checkpoint, "
                         "whose prompts share the state as a prefix")
    if hasattr(m, "merge_and_unload"):  # d-only 的档没有 LoRA, m 就是基模本身
        m = m.merge_and_unload()
    return Engine(m, tok, d_ids, context_label=context_label, type_marker=bool(cfg.get("type_marker", False)),
                  context_marker=bool(cfg.get("context_marker", False)), max_tokens=max_tokens, max_batch_tokens=max_batch_tokens,
                  candidate_prefix_cache=candidate_prefix_cache)
