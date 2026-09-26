"""推理引擎: 一份训好的存档, 回答一个 state 上的一组 API 问题, 给出每道题在它菜单各行上的分布.

一个请求怎么跑:
  1. 每道题照 serve.menus 排成菜单, 按训练时的 context-first 模板切成两段: state 那段所有题共用, 问题那段各题自己的
  2. state 那段前向一次, 得到它的 KV cache (core.cache.prefix_cache)
  3. 问题按长度排好分组, 每组的分支接在 cache 后面一次前向 (core.cache.branch_logits).
     一组的 cache 是 组员数 × (state + 组内最长的问题) 个 token, 不超过 max_batch_tokens; 一题超了就单独一组
  4. 每道题在自己菜单的 k 个 D 码上做 softmax. 与评估 (evaluation.scoring) 读的是同一个分布
state 加上最长的那道题超过 max_tokens 时整个请求拒收, 不碰模型.
GPU 同一时刻只跑一个请求 (一把锁), 并发的请求排队.

load_engine 从基模 + 存档搭引擎: LoRA 照档里记的形状挂上再合并进权重 (部署形态, 比旁路挂法快),
档里记的 type_marker 照搬. menu-first 训的档没有可共享的 state 前缀, 拒收.
"""

from __future__ import annotations

import json
import pathlib
import threading
from dataclasses import dataclass

import torch

from decidophobia.core.cache import branch_logits, prefix_cache
from decidophobia.core.checkpoint import prepare_from_checkpoint
from decidophobia.core.prompt import split_prompt
from decidophobia.core.tokens import install_d_tokens, install_type_tokens
from decidophobia.serve.menus import to_example

LAYOUT = "context-first"  # state 在前才有可共享的前缀


class RequestTooLong(ValueError):
    """state 加最长的问题超过引擎的 max_tokens."""


@dataclass(frozen=True)
class Evaluation:
    probs: dict[str, list[float]]  # 问题 id -> 菜单各行的概率, 行序见 serve.menus
    input_tokens: int  # state 一次 + 每道题各自那段


class Engine:
    def __init__(self, lm, tok, d_ids: list[int], context_label: str = "State", type_marker: bool = False,
                 max_tokens: int = 8192, max_batch_tokens: int = 16384):
        self.lm = lm.eval()
        self.tok = tok
        self.d_ids = d_ids
        self.context_label = context_label
        self.type_marker = type_marker
        self.max_tokens = max_tokens
        self.max_batch_tokens = max_batch_tokens
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        self._lock = threading.Lock()

    def _encode(self, s: str) -> list[int]:
        return self.tok.encode(s, add_special_tokens=False)

    def evaluate(self, state, questions: dict) -> Evaluation:
        """questions: {问题 id: serve.api 的 Noul / Choice / Score}. 返回的 probs 与 questions 同序."""
        exs = {qid: to_example(q, state, self.context_label) for qid, q in questions.items()}
        parts = {qid: split_prompt(ex, LAYOUT, self.type_marker) for qid, ex in exs.items()}
        ctx = self._encode(next(iter(parts.values()))[0])
        segs = {qid: self._encode(seg) for qid, (_, seg) in parts.items()}
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


def load_engine(checkpoint, base_model, context_label: str = "State", device: str = "cuda",
                dtype: torch.dtype = torch.bfloat16, max_tokens: int = 8192, max_batch_tokens: int = 16384) -> Engine:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    d_ids = install_d_tokens(tok)
    train_ids = d_ids + install_type_tokens(tok)
    lm = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype).to(device)
    m, cfg = prepare_from_checkpoint(lm, train_ids, checkpoint)
    layout = cfg.get("layout", LAYOUT)
    if layout != LAYOUT:
        raise ValueError(f"{checkpoint} was trained {layout}; the server needs a {LAYOUT} checkpoint, "
                         "whose prompts share the state as a prefix")
    if hasattr(m, "merge_and_unload"):  # d-only 的档没有 LoRA, m 就是基模本身
        m = m.merge_and_unload()
    return Engine(m, tok, d_ids, context_label=context_label, type_marker=bool(cfg.get("type_marker", False)),
                  max_tokens=max_tokens, max_batch_tokens=max_batch_tokens)
