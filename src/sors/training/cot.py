"""训练时的计算过程 (cot): 只在训练时让模型学着写, 推理永远不写.

每道带 MenuExample.cot 的题, 除了照常走决策打分, 再走一遍语言模型:
  序列 = 决策提示原文 (到 "Answer:" 为止, 与决策那一遍逐 token 相同) + " " + cot + <eos>
  损失 = 只在 cot 与 <eos> 上的下一个 token 交叉熵; 每条序列内取平均, 再对序列取平均
  前向 = 主干加冻住的词表输出层, 不挂决策层 (决策层把选项写回每个位置, 挂上会让计算过程依赖选项)
cot 的第一个 token 由 "Answer:" 那个位置预测, 而决策读的也是这个位置: 「此处就要知道怎么算」的压力落在决策读取点上.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sors.core.decision_batch import _minimal_batch, _pad
from sors.core.prompt import encode_prompts, prompt_pieces


def _prompts(examples, tok, d_ids, layout, max_length, type_marker, context_marker, architecture):
    if architecture == "minimal":
        return [ids for ids, _ in _minimal_batch(examples, tok, d_ids, layout, type_marker, context_marker,
                                                 max_length, True)]
    if architecture != "slots":
        raise ValueError(f"the cot loss supports the minimal and slots architectures, not {architecture!r}")
    from sors.core.batch import fit
    return [fit(e, max_length, None) for e in
            encode_prompts(tok, [sum(prompt_pieces(ex, layout, type_marker, context_marker), []) for ex in examples])]


def cot_batch(examples, tok, d_ids, layout, max_length, type_marker, context_marker, *, paired: bool,
              architecture: str = "minimal") -> dict[str, torch.Tensor]:
    """语言模型那一遍的张量 cot_input_ids / cot_attention_mask / cot_labels (左填充, 标签 -100 = 不算).
    paired: 批是 [a, a', b, b', ...] 的一致性配对, 每道题只取第一份. 批里没有 cot 时返回 {}."""
    picked = [ex for i, ex in enumerate(examples) if ex.cot and (not paired or i % 2 == 0)]
    if not picked:
        return {}
    prompts = _prompts(picked, tok, d_ids, layout, max_length, type_marker, context_marker, architecture)
    work = tok([" " + ex.cot for ex in picked], add_special_tokens=False)["input_ids"]
    eos = [tok.eos_token_id] if tok.eos_token_id is not None else []
    seqs = [p + w + eos for p, w in zip(prompts, work)]
    labels = [[-100] * len(p) + w + eos for p, w in zip(prompts, work)]
    ids, mask = _pad(seqs, tok.pad_token_id if tok.pad_token_id is not None else 0)
    lab, _ = _pad(labels, -100)
    return {"cot_input_ids": ids, "cot_attention_mask": mask, "cot_labels": lab}


def _parts(m):
    base = getattr(m, "base", m)  # DecisionModel 包着 CausalLM; slots 架构就是 CausalLM (或 peft 包装)
    raw = base.get_base_model() if hasattr(base, "get_base_model") else base
    return raw.model, raw.lm_head


def cot_lm_loss(m, b: dict) -> torch.Tensor:
    """cot 上的语言模型损失: 每条序列内对 cot token 取平均, 再对序列取平均. 决策层不参与."""
    decoder, head = _parts(m)
    ids, mask, labels = b["cot_input_ids"], b["cot_attention_mask"], b["cot_labels"]
    positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
    h = decoder(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False).last_hidden_state
    target = labels[:, 1:]
    sel = target != -100
    logits = head(h[:, :-1][sel]).float()
    tok_loss = F.cross_entropy(logits, target[sel], reduction="none")
    rows = sel.nonzero()[:, 0]
    per_seq = torch.zeros(ids.shape[0], device=tok_loss.device).index_add_(0, rows, tok_loss)
    counts = sel.sum(1).clamp_min(1).to(per_seq.dtype)
    return (per_seq / counts).mean()
