"""蒸馏式内化 (candidate + 打包前向): 让「不写计算过程」的学生读出, 贴近「读过计算过程」的老师读出.

一次打包前向里每道带计算过程 (MenuExample.trace) 的题有两套分支:
  学生 = 前缀 + 选项 (推理时的形式)        老师 = 前缀 + 计算过程 + 选项
计算过程只是老师读的输入, 任何位置都不算下一个词的损失. 监督全落在选项读出的 H 上:
  - 学生照常用答案训练 (调用方的 menu 损失, 用 student_logits);
  - 老师也用答案训练 (teacher_logits, 只含带计算过程的题), 保持老师够强;
  - distill = 1 - cos(学生 H, stop_grad(老师 H)), 只在带计算过程的题的各选项上取平均.
"""

import torch
import torch.nn.functional as F


def distill_outputs(m, batch: dict) -> dict:
    """返回 student_logits (B, V), teacher_logits (B_t, V) 与 teacher_slots / teacher_rows (带计算过程的行),
    distill (标量), teacher_h (detach 后的老师读出, 只读)."""
    from sors.core.packed import packed_candidate_hidden, supports_packed

    if m.decision_config.kind != "candidate" or m.candidate_prefix_cache != "packed":
        raise ValueError("distillation needs the candidate architecture with the packed prefix cache")
    if not supports_packed(m.decoder):
        raise ValueError("packed candidate encoding supports Qwen3.5 text backbones only")
    if "trace_input_ids" not in batch:
        raise ValueError("distillation needs a batch collated with traces=True")
    slots = batch["slot_ids"]
    valid = slots >= 0
    has = batch["trace_attention_mask"].bool().any(1)
    with m._forward_lock:
        student_h, teacher_h = packed_candidate_hidden(
            m.decoder, batch, valid, checkpointed=m.checkpoint_forward and m.training and torch.is_grad_enabled(),
            traces=True)
    dtype = m.option_norm.weight.dtype
    student_h, teacher_h = student_h.to(dtype), teacher_h.to(dtype)
    student_logits = m._candidate_output(student_h, slots)
    out = {"student_logits": student_logits, "teacher_rows": has.nonzero().flatten()}
    if teacher_h.shape[0] == 0:
        zero = student_h.sum() * 0
        out.update(teacher_logits=student_logits[:0], teacher_slots=slots[:0], distill=zero,
                   teacher_h=teacher_h.detach())
        return out
    teacher_slots = slots[has]
    out["teacher_logits"] = m._candidate_output(teacher_h, teacher_slots)
    out["teacher_slots"] = teacher_slots
    # 学生里与老师对应的那几行: valid.nonzero() 顺序中属于带计算过程的例子的项
    rows = valid.nonzero()[:, 0]
    pick = has[rows].nonzero().flatten()
    target = teacher_h.detach()
    out["distill"] = (1 - F.cosine_similarity(student_h[pick].float(), target.float(), dim=-1)).mean()
    out["teacher_h"] = target
    return out
