"""评估打分: 一批 MenuExample 过模型, 读出每道题在菜单各行上的概率, 再汇总成指标.

score_examples 逐题打分不汇总; evaluate 在一个评估集上报正确率、NLL、校准与格式遵从;
consistency_eval 把同一批题的几种排法各打一次分, 按描述对齐比. 训练循环 (decidophobia.training.loop) 的评估点与评估脚本都用它们.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from decidophobia.core.batch import collate
from decidophobia.core.menu import MenuExample
from decidophobia.core.model import last_logits
from decidophobia.evaluation.metrics import (answer_mass_summary, binary_summary, by_gold_slot, first_two_slots, menu_size_summary,
                                             pass_consistency, summarize)
from decidophobia.training.loss import answer_mass, gather_slot_logits, vocab_cross_entropy


@dataclass
class EvalSet:
    examples: list[MenuExample]
    batch_size: int = 16
    pos_class: int | None = None  # 二元数据集给正类 id, 就多报 auroc / pos_rate / brier_binary


@torch.no_grad()
def score_examples(m, tok, d_ids, examples: list[MenuExample], batch_size: int, k_max: int, max_length: int,
                   layout: str, type_marker: bool = False, context_marker: bool = False) -> dict[str, list]:
    """逐题打分, 不汇总.
      q          每道题在自己菜单 k 个槽上的概率 (位置空间, 长 k_max, 菜单之外补 0)
      gold       正确选项的位置
      vocab_ce   全词表交叉熵, 与 --loss vocab 的训练 loss 同一个式子
      m_answer / m_offmenu / top1_in   全词表下的三个格式读数, 见 loss.answer_mass
    """
    was_training = m.training
    m.eval()
    dev = next(m.parameters()).device
    out = {"q": [], "gold": [], "vocab_ce": [], "m_answer": [], "m_offmenu": [], "top1_in": []}
    for s in range(0, len(examples), batch_size):
        kind = getattr(getattr(m, "decision_config", None), "kind", "slots")
        b = collate(examples[s : s + batch_size], tok, d_ids, k_max, layout, max_length, type_marker, context_marker,
                    architecture=kind)
        b = {k: v.to(dev) for k, v in b.items()}
        logits = m.forward_batch(b) if kind != "slots" else last_logits(m, b["input_ids"], b["attention_mask"])
        q = torch.softmax(gather_slot_logits(logits, b["slot_ids"]), dim=-1)  # pad 槽 exp(-inf)=0
        ma, off, top1 = answer_mass(logits, b["slot_ids"], d_ids)
        out["q"].extend(q.cpu().tolist())
        out["gold"].extend(b["gold"].tolist())
        out["vocab_ce"].extend(vocab_cross_entropy(logits, b["slot_ids"], b["gold"], reduction="none").cpu().tolist())
        out["m_answer"].extend(ma.cpu().tolist())
        out["m_offmenu"].extend(off.cpu().tolist())
        out["top1_in"].extend(top1.cpu().tolist())
    if was_training:
        m.train()
    return out


def evaluate(m, tok, d_ids, es: EvalSet, k_max: int, max_length: int, layout: str, type_marker: bool = False,
             context_marker: bool = False) -> dict:
    """summarize() 那组指标 (位置空间), 二元集再加 binary_summary (类空间). 概率只在各自菜单的 k 个槽上归一.
    另报格式遵从 (answer_mass_summary): 全词表下有多少概率落在菜单的槽上, 与 baseline 脚本的 m_answer 同一个量.
    vocab_ce 是评估集上的 loss, 分母是整个词表, 与 train/loss (--loss vocab) 直接可比;
    nll 只在菜单上归一, 与 ece / brier / accuracy 同一个分布, 也与旧 run 的曲线同一个量."""
    s = score_examples(m, tok, d_ids, es.examples, es.batch_size, k_max, max_length, layout, type_marker, context_marker)
    Q, Y = s["q"], s["gold"]
    out = summarize(Q, Y)
    out["vocab_ce"] = sum(s["vocab_ce"]) / len(Y)
    if es.pos_class is not None:
        out.update(binary_summary(Q, es.examples, es.pos_class))
    out.update(answer_mass_summary(s["m_answer"], s["m_offmenu"], s["top1_in"]))
    out.update(menu_size_summary([len(ex.options) for ex in es.examples]))
    out.update(first_two_slots(Q, Y))
    out["n"] = len(Y)
    out["by_gold_slot"] = by_gold_slot(Q, Y)
    return out


def consistency_eval(m, tok, d_ids, passes: list[list[MenuExample]], batch_size: int, k_max: int, max_length: int,
                     layout: str, type_marker: bool = False, context_marker: bool = False) -> dict:
    """每一份各打一次分, 再按描述对齐比 (metrics.pass_consistency): accuracy / agree / js."""
    qs = [score_examples(m, tok, d_ids, exs, batch_size, k_max, max_length, layout, type_marker, context_marker)["q"]
          for exs in passes]
    return pass_consistency(qs, passes)
