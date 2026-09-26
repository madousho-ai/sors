"""BoolQ 适配: google/boolq, passage 是上下文、question 是逐条问句、答案 yes/no.

与 scripts/baseline-boolq.py 同一份数据 (validation 3270 条, yes 占 0.6217).
类 id: 0 = no, 1 = yes. 菜单里显示 "no" / "yes", 位置由 compose_menu 打乱.
"""

from __future__ import annotations

from datasets import load_dataset

from decidophobia.core.menu import LabeledSet

NAMES = {0: "no", 1: "yes"}


def _mk(ds) -> LabeledSet:
    return LabeledSet(
        queries=[p.strip() for p in ds["passage"]],
        labels=[1 if a else 0 for a in ds["answer"]],
        names=NAMES,
        context_label="Passage",
        questions=[q.strip().rstrip("?") + "?" for q in ds["question"]],
        qtype="bool",
    )


def load_boolq() -> tuple[LabeledSet, LabeledSet]:
    """返回 (train 9427, validation 3270)."""
    return _mk(load_dataset("google/boolq", split="train")), _mk(load_dataset("google/boolq", split="validation"))
