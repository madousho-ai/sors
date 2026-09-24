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


def by_gold_slot(q: list[list[float]], y: list[int], width: int = 10) -> dict[str, dict[str, float]]:
    """按正确答案所在的槽 (D0, D1, ...) 每 width 个分一档, 各档单独报 n / accuracy / top5 / nll / conf_mean.
    看得出哪一段码训到了、哪一段是死的. 没有题的档不出现. 档内比较要在同一菜单长度下才公平."""
    groups: dict[int, list[int]] = {}
    for i, t in enumerate(y):
        groups.setdefault(t // width, []).append(i)
    out = {}
    for b in sorted(groups):
        qs = [q[i] for i in groups[b]]
        ys = [y[i] for i in groups[b]]
        out[f"{b * width}-{b * width + width - 1}"] = {
            "n": len(ys),
            "accuracy": topk_accuracy(qs, ys, k=1),
            "top5_accuracy": topk_accuracy(qs, ys, k=5),
            "nll": nll_multiclass(qs, ys),
            "conf_mean": sum(max(r) for r in qs) / len(qs),
        }
    return out


def first_two_slots(q: list[list[float]], y: list[int]) -> dict[str, float]:
    """前两格 (D0 / D1) 吸走了多少: 选在前两格的比例 / 正确答案在前两格的比例 / 前两格的平均概率.
    no / yes 题的答案永远在 D0 / D1; 训练里掺了它们, 菜单题的预测若被拉向前两格, pred 会高出 gold."""
    return {
        "pred_d01_rate": sum(_argsort_desc(row)[0] < 2 for row in q) / len(q),
        "gold_d01_rate": sum(t < 2 for t in y) / len(y),
        "q_d01_mean": sum(sum(row[:2]) for row in q) / len(q),
    }


def menu_size_summary(ks: list[int]) -> dict[str, float]:
    """评估集里每道题的菜单长度."""
    return {"k_min": min(ks), "k_max": max(ks), "k_mean": sum(ks) / len(ks)}


def consistency(base_q: list[list[float]], base_exs, var_q: list[list[float]], var_exs,
                eps: float = NLL_EPS) -> dict[str, float]:
    """同一批题的变体菜单 (换行 / 换码 / 删选项) 与原菜单比. 按类 id 对齐, 只在变体菜单上的那些描述上比:
    原菜单的分布先限制到这些描述并重新归一 —— 读出与行、码、长度无关时, 删掉的又只是没选的选项, 两边应当相同.
    q 的每行按位置排, 可以带菜单之外补的 0 列.

      flip_rate        原菜单选中的描述还在变体菜单上的题里, 变体选了别的描述的比例
      n_comparable     flip_rate 的分母; 只换行换码时等于 n
      tv_mean          两个分布的总变差距离, 逐题平均
      gold_logp_drift  正确描述的对数概率之差的绝对值, 逐题平均
    """
    flips = comparable = 0
    tv = drift = 0.0
    for bq, be, vq, ve in zip(base_q, base_exs, var_q, var_exs):
        base = dict(zip(be.options, bq))
        var = dict(zip(ve.options, vq))
        if ve.label != be.label or not var.keys() <= base.keys():
            raise ValueError(f"variant {ve.options} (gold {ve.label}) is not a subset of {be.options} (gold {be.label})")
        z = sum(base[c] for c in var)
        restricted = {c: base[c] / z for c in var}
        pick = max(base, key=base.get)
        if pick in var:
            comparable += 1
            flips += max(var, key=var.get) != pick
        tv += 0.5 * sum(abs(var[c] - restricted[c]) for c in var)
        drift += abs(math.log(max(var[ve.label], eps)) - math.log(max(restricted[ve.label], eps)))
    n = len(base_exs)
    return {"n": n, "n_comparable": comparable, "flip_rate": flips / comparable if comparable else None,
            "tv_mean": tv / n, "gold_logp_drift": drift / n}


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
