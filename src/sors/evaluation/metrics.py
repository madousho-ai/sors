"""多类指标. 与 scripts/baseline-banking77.py 里的实现相同, 期望值由同一组手算测试钉住."""

from __future__ import annotations

import math

from sors.core.menu import row_alignment

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


def _top(row: list[float], k: int) -> int:
    """菜单前 k 行里概率最大的那一行 (菜单之外补的 0 不参与)."""
    return max(range(k), key=row.__getitem__)


def decoy_summary(q: list[list[float]], examples) -> dict:
    """诱饵读数, 只看带诱饵 (ex.decoy) 的题; 一道都没有就返回空 dict.
      n_decoy          带诱饵的题数
      decoy_rate       首选正是诱饵的比例
      decoy_of_errors  答错的题里选中诱饵的比例. 高 = 错在跟着表面走; 接近 decoy_chance = 错得分散
      decoy_chance     同一批错题若在错答案里随便挑, 挑中诱饵的期望比例 (逐题 1/(k-1) 平均)
    带诱饵的题全答对时后两个没有分母, 给 None."""
    rows = [(row, ex) for row, ex in zip(q, examples) if ex.decoy is not None]
    if not rows:
        return {}
    picks = [(ex.options[_top(row, len(ex.options))], ex) for row, ex in rows]
    on_decoy = sum(p == ex.decoy for p, ex in picks)
    wrong = [ex for p, ex in picks if p != ex.label]
    return {
        "n_decoy": len(rows),
        "decoy_rate": on_decoy / len(rows),
        "decoy_of_errors": on_decoy / len(wrong) if wrong else None,
        "decoy_chance": sum(1 / (len(ex.options) - 1) for ex in wrong) / len(wrong) if wrong else None,
    }


def by_family(q: list[list[float]], examples) -> dict[str, dict]:
    """按题目类别 (ex.family) 分开报 n / accuracy / nll / conf_mean, 带诱饵的类别再加 decoy_summary.
    没有 family 的题不进任何一档; 档按 family 首次出现的顺序排."""
    groups: dict[str, list[int]] = {}
    for i, ex in enumerate(examples):
        if ex.family is not None:
            groups.setdefault(ex.family, []).append(i)
    out = {}
    for fam, idx in groups.items():
        qs, exs = [q[i] for i in idx], [examples[i] for i in idx]
        ys = [ex.gold_idx for ex in exs]
        out[fam] = {
            "n": len(idx),
            "accuracy": sum(_top(row, len(ex.options)) == ex.gold_idx for row, ex in zip(qs, exs)) / len(idx),
            "nll": nll_multiclass(qs, ys),
            "conf_mean": sum(max(row) for row in qs) / len(idx),
            **decoy_summary(qs, exs),
        }
    return out


def pass_by_family(qs: list[list[list[float]]], passes: list[list]) -> dict[str, dict]:
    """同一批题的几种排法 (menu.arrangements) 各算一次 by_family, 逐项取平均. 首选按描述 (类 id) 认,
    与它落在哪一行无关. n / n_decoy 每份相同, 照抄; 某份某项是 None (没有分母) 的, 只在有值的那几份上平均."""
    per = [by_family(q, exs) for q, exs in zip(qs, passes)]
    out = {}
    for fam, first in per[0].items():
        out[fam] = {}
        for key in first:
            vals = [p[fam][key] for p in per if p[fam][key] is not None]
            out[fam][key] = (first[key] if key in ("n", "n_decoy") else sum(vals) / len(vals)) if vals else None
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


def _entropy(p: list[float]) -> float:
    return -sum(x * math.log(x) for x in p if x > 0)


def pass_consistency(qs: list[list[list[float]]], passes: list[list]) -> dict[str, float]:
    """同一批题的几种排法 (menu.arrangements) 放在一起比. qs[p][i] 是第 p 份第 i 道题在菜单各行上的概率 (按位置,
    可以带菜单之外补的 0 列), passes[p][i] 是那一份的菜单. 各份按描述对齐 (menu.row_alignment) 到第 0 份的顺序.

      accuracy  每份每题首选是不是正确描述, 全部平均. 首选只能靠读描述选对, 压平分布或乱猜刷不出来
      agree     各份首选都是同一条描述的题占多少
      js        各份分布的 Jensen-Shannon 散度 H(平均分布) − 平均 H, 逐题平均. 两份时就是 loss.consistency_js,
                0 = 排法完全不影响给每条描述的概率; 上限 ln(份数)
    """
    P, n = len(passes), len(passes[0])
    correct = agree = 0
    js = 0.0
    for i in range(n):
        base = passes[0][i]
        dists, picks = [], set()
        for p in range(P):
            ex, q = passes[p][i], qs[p][i][: len(passes[p][i].options)]
            top = max(range(len(q)), key=q.__getitem__)
            correct += top == ex.gold_idx
            picks.add(ex.options[top])
            dists.append([q[j] for j in row_alignment(base, ex)])
        agree += len(picks) == 1
        mean = [sum(col) / P for col in zip(*dists)]
        js += _entropy(mean) - sum(_entropy(d) for d in dists) / P
    return {"n": n, "passes": P, "accuracy": correct / (n * P), "agree": agree / n, "js": js / n}


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
