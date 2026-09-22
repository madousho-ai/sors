"""多类指标. 与 scripts/baseline-banking77.py 里的实现相同, 期望值由同一组手算测试钉住."""

from __future__ import annotations

import math

NLL_EPS = 1e-12


def _argsort_desc(row: list[float]) -> list[int]:
    return sorted(range(len(row)), key=lambda k: -row[k])


def topk_accuracy(q: list[list[float]], y: list[int], k: int) -> float:
    hits = sum(1 for row, t in zip(q, y) if t in _argsort_desc(row)[:k])
    return hits / len(y)


def nll_multiclass(q: list[list[float]], y: list[int], eps: float = NLL_EPS) -> float:
    return -sum(math.log(max(row[t], eps)) for row, t in zip(q, y)) / len(y)


def count_saturated_multiclass(q: list[list[float]], y: list[int], eps: float = NLL_EPS) -> int:
    return sum(1 for row, t in zip(q, y) if row[t] < eps)


def brier_multiclass(q: list[list[float]], y: list[int]) -> float:
    """mean_i sum_k (q_ik - y_ik)^2, 无 1/K."""
    total = 0.0
    for row, t in zip(q, y):
        total += sum((v - (1.0 if k == t else 0.0)) ** 2 for k, v in enumerate(row))
    return total / len(y)


def ece(conf: list[float], correct: list[bool], n_bins: int = 10) -> float:
    n = len(conf)
    sums = [0.0] * n_bins
    hits = [0] * n_bins
    counts = [0] * n_bins
    for c, ok in zip(conf, correct):
        b = min(int(c * n_bins), n_bins - 1)
        sums[b] += c
        hits[b] += 1 if ok else 0
        counts[b] += 1
    total = 0.0
    for b in range(n_bins):
        if counts[b] == 0:
            continue
        total += counts[b] / n * abs(sums[b] / counts[b] - hits[b] / counts[b])
    return total


def ece_multiclass(q: list[list[float]], y: list[int], n_bins: int = 10) -> float:
    conf = [max(row) for row in q]
    correct = [_argsort_desc(row)[0] == t for row, t in zip(q, y)]
    return ece(conf, correct, n_bins=n_bins)


def summarize(q: list[list[float]], y: list[int], n_bins: int = 10) -> dict[str, float]:
    """一次算全: accuracy / top5 / nll / n_saturated / brier / ece / conf_mean."""
    return {
        "accuracy": topk_accuracy(q, y, k=1),
        "top5_accuracy": topk_accuracy(q, y, k=5),
        "nll": nll_multiclass(q, y),
        "n_saturated": count_saturated_multiclass(q, y),
        "brier": brier_multiclass(q, y),
        "ece": ece_multiclass(q, y, n_bins=n_bins),
        "conf_mean": sum(max(r) for r in q) / len(q),
    }


def _percentile(xs: list[float], pct: float) -> float:
    """线性插值, 与 numpy.percentile 的默认方法相同 (baseline 脚本用的是它)."""
    s = sorted(xs)
    pos = pct / 100 * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def answer_mass_summary(m: list[float], off: list[float], top1_in: list[bool]) -> dict[str, float]:
    """loss.answer_mass 三个逐条读数的汇总. m_answer_* 与 baseline 脚本同名同算法, 基模与训练后并排比."""
    n = len(m)
    return {
        "m_answer_mean": sum(m) / n,
        "m_answer_p05": _percentile(m, 5),
        "m_answer_p95": _percentile(m, 95),
        "m_offmenu_mean": sum(off) / n,
        "top1_in_menu_rate": sum(top1_in) / n,
    }


# --------------------------------------------------------------------------
# 二元 (BoolQ): 位置空间的 q 映回类空间, 报 AUROC / 正类率 / 二元 Brier
# --------------------------------------------------------------------------


def auroc(scores: list[float], labels: list[int]) -> float | None:
    """Mann-Whitney: 正例分数高于负例的 (正, 负) 对占比, 平局记半分. 只有一类时返回 None."""
    pos = [s for s, t in zip(scores, labels) if t == 1]
    neg = [s for s, t in zip(scores, labels) if t == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def binary_summary(q: list[list[float]], examples, pos_class: int) -> dict[str, float | None]:
    """每条样本取正类所在位置的概率 P(pos), 与 label == pos_class 对齐.
    pos_rate: 预测为正 (P(pos) >= 0.5) 的比例; label_pos_rate: 标签里正类的比例 —— 两者之差是偏置."""
    p_pos = [row[ex.options.index(pos_class)] for row, ex in zip(q, examples)]
    y = [1 if ex.label == pos_class else 0 for ex in examples]
    n = len(y)
    return {
        "p_pos_mean": sum(p_pos) / n,
        "pos_rate": sum(1 for p in p_pos if p >= 0.5) / n,
        "label_pos_rate": sum(y) / n,
        "auroc": auroc(p_pos, y),
        "brier_binary": sum((p - t) ** 2 for p, t in zip(p_pos, y)) / n,
    }
