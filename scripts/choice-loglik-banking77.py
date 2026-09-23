#!/usr/bin/env python
"""Banking77 上检验「逐选项续写似然」采集 Base 决策分布的可靠性.

被检验的方法 (外部提案): 提示 = 用户句 + 问句 + 全部选项列表 + 'Answer:';
把每个选项名字接在后面, teacher-forced 一次前向读出它每个 token 的 log P, 累加成分数,
K 个分数过 restricted softmax 当 teacher 分布.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# 纯函数 (tests/test_choice_loglik.py 覆盖)
# --------------------------------------------------------------------------

QUESTION = "Which option best describes the message?"


def render_prefix(query: str, choices: list[str], with_menu: bool = True) -> str:
    """用户句 + 问句 + 选项列表 + 'Answer:'. 选项续写是 ' 名字\\n', 前导空格在续写那边.

    with_menu=False 是对照: 模型看不到有哪些选项, 只能靠名字本身的无条件似然排序.
    """
    head = f"Customer message: {query}\n\nQuestion: {QUESTION}\n"
    if not with_menu:
        return head + "Answer:"
    return head + "Choices:\n" + "\n".join(f"- {c}" for c in choices) + "\n\nAnswer:"


def aggregate(lp: np.ndarray, how: str, terminator: bool) -> np.ndarray:
    """(N, K, L) 逐 token log P -> (N, K) 分数. 无效位是 NaN, 每行有效位连续靠左.

    每个选项续写的最后一个有效 token 是终止符 ('\\n'): 它问的是「名字到这里说完了吗」.
    terminator=False 把它去掉, 只剩名字本身的 token.
    how: 'sum' 累加; 'avg' 除以计入的 token 数.
    """
    if how not in ("sum", "avg"):
        raise ValueError(f"unknown aggregation {how!r}")
    valid = ~np.isnan(lp)
    if not terminator:
        n = valid.sum(-1, keepdims=True)
        pos = np.arange(lp.shape[-1])
        valid = valid & (pos < n - 1)
    x = np.where(valid, lp, 0.0)
    s = x.sum(-1)
    if how == "avg":
        s = s / valid.sum(-1)
    return s


def restricted_softmax(scores: np.ndarray, T: float = 1.0) -> np.ndarray:
    """(N, K) 分数在最后一维上 softmax, 先除以温度 T."""
    z = scores / T
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def position_mass(q_cls: np.ndarray, order: np.ndarray) -> np.ndarray:
    """(N, K) 按类排的概率 -> (K,) 每个菜单行平均拿到的概率. 无偏好时每行 1/K."""
    rows = np.arange(q_cls.shape[0])[:, None]
    return q_cls[rows, order].mean(0)


def entropy_bits(q: np.ndarray) -> np.ndarray:
    """(N, K) -> (N,) 以 bit 计的熵, 0·log0 记 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(q > 0, q * np.log2(q), 0.0)
    return -t.sum(-1)


def pairwise_agreement(qs: list[np.ndarray]) -> dict[str, float]:
    """几套菜单顺序下的分布 (都已映回类空间), 两两比: argmax 一致率与总变差距离的平均."""
    agree, tv = [], []
    for i in range(len(qs)):
        for j in range(i + 1, len(qs)):
            agree.append(float((qs[i].argmax(-1) == qs[j].argmax(-1)).mean()))
            tv.append(float(0.5 * np.abs(qs[i] - qs[j]).sum(-1).mean()))
    return {"argmax_agree": float(np.mean(agree)), "tv_mean": float(np.mean(tv))}


def _nll_at(scores: np.ndarray, y: np.ndarray, T: float) -> float:
    z = scores / T
    z = z - z.max(-1, keepdims=True)
    lse = np.log(np.exp(z).sum(-1))
    return float((lse - z[np.arange(len(y)), y]).mean())


def fit_temperature(scores: np.ndarray, y: np.ndarray, lo: float = 0.02, hi: float = 100.0) -> float:
    """在 [lo, hi] 上找使 NLL 最小的 T. 对数网格粗搜, 再在最优格点两侧做黄金分割细化.
    NLL 这种光滑的 proper scoring rule 当目标, ECE 的分箱会让目标变成阶梯."""
    grid = np.geomspace(lo, hi, 200)
    vals = [_nll_at(scores, y, t) for t in grid]
    k = int(np.argmin(vals))
    a, b = np.log(grid[max(k - 1, 0)]), np.log(grid[min(k + 1, len(grid) - 1)])
    g = (np.sqrt(5) - 1) / 2
    for _ in range(60):
        c, d = b - g * (b - a), a + g * (b - a)
        if _nll_at(scores, y, np.exp(c)) < _nll_at(scores, y, np.exp(d)):
            b = d
        else:
            a = c
    return float(np.exp((a + b) / 2))


def heldout_temperature(scores: np.ndarray, y: np.ndarray, folds: int = 2) -> tuple[np.ndarray, list[float]]:
    """按下标对 folds 取模分折; 每折的分布用在其余折上拟合的 T 算. 返回 (q, 每折用到的 T)."""
    fold = np.arange(len(y)) % folds
    q = np.empty_like(scores, dtype=np.float64)
    temps = []
    for f in range(folds):
        t = fit_temperature(scores[fold != f], y[fold != f])
        temps.append(t)
        q[fold == f] = restricted_softmax(scores[fold == f], T=t)
    return q, temps


def length_profile(q: np.ndarray, y: np.ndarray, name_len: np.ndarray) -> list[dict]:
    """按类名 token 数分组. 每组: 几个类、标签落在这组的比例、概率质量落在这组的比例、argmax 落在这组的比例.
    质量占比远超标签占比 = 这个长度的名字被系统性偏爱."""
    pred = q.argmax(-1)
    out = []
    for L in sorted(set(name_len.tolist())):
        members = name_len == L
        out.append({
            "len": int(L),
            "n_classes": int(members.sum()),
            "label_share": float(members[y].mean()),
            "mass_share": float(q[:, members].sum(-1).mean()),
            "pred_share": float(members[pred].mean()),
        })
    return out
