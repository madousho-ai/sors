#!/usr/bin/env python
"""Banking77 上检验「逐选项续写似然」采集 Base 决策分布的可靠性.

被检验的方法 (外部提案): 提示 = 用户句 + 问句 + 全部选项列表 + 'Answer:';
把每个选项名字接在后面, teacher-forced 一次前向读出它每个 token 的 log P, 累加成分数,
K 个分数过 restricted softmax 当 teacher 分布.

每条测试句跑这几种条件, 77 个选项全部打分:
  perm0..perm{P-1}  菜单顺序不同的同一份提示. perm0 按名字字母序, 其余按 seed 打乱; 一个顺序全数据集共用
  bare              提示里不给菜单. 与 perm 的差 = 模型读菜单的作用
  null (每个 perm)   用户句换成 '' / 'N/A' / '[MASK]', 给出每个选项的无条件先验, 用于 PMI 校正
每个选项的续写是 ' 名字\\n', 逐 token 存 log P; 最后一位 (终止符) 取 {\\n, \\n\\n, EOS} 的概率和.
聚合方式四种 (sum / avg × 是否计入终止符) 外加只看首 token, 全部离线从同一份逐 token 数组算.

  OMP_NUM_THREADS=2 PYTHONPATH=src HF_HUB_OFFLINE=1 .venv/bin/python scripts/choice-loglik-banking77.py --limit 50

产物: results/<ts>-choice-loglik-banking77-<model>-<dtype>-n<N>.{json,npz,txt}
  json 指标 + 自检 + 元数据; npz 逐 token log P (条件 × N × 77 × L); txt 几条样本的逐 token 明细.
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


# --------------------------------------------------------------------------
# GPU: 前缀一次前向, 选项分块 teacher-forced (tests/test_choice_loglik_gpu.py 覆盖)
# --------------------------------------------------------------------------


def stop_token_ids(tok) -> list[int]:
    """「答案到这里说完了」的 token: 换行、双换行、EOS. 终止符位置取这几个 token 的概率之和."""
    ids = {tok.encode("\n", add_special_tokens=False)[0], tok.encode("\n\n", add_special_tokens=False)[0]}
    ids.add(tok.eos_token_id)
    return sorted(ids)


def score_continuations(lm, prefix_ids: list[int], conts: list[list[int]], stop_ids: list[int], chunk: int):
    """前缀前向一次得 KV cache; 选项每 chunk 个一组, 组内首尾相接排成一条序列接在 cache 后面前向.

    打包 (tree attention): 组内每个续写只看得到前缀和它自己已经出现的 token —— 用 (1, 1, T, n_ctx+T)
    的布尔 4D 掩码表达, 位置编号各自从 n_ctx 起算. 于是前缀的 KV 只存一份, 无填充.
    每个续写 c 喂进去的是 c[:-1]: 序列里第 j 个 token 的 logits 预测 c[j+1]; c[0] 由前缀最后位置预测.
    c 的最后一个 token 是终止符, 那一位取 stop_ids 上的 logsumexp.
    返回 (lp, first): lp 是 (K, Lmax) 的 np.float32, 填充位 NaN; first 是前缀最后位置的全词表 logits.
    """
    import copy

    import torch
    from transformers import DynamicCache

    dev = lm.device
    with torch.no_grad():
        cache = DynamicCache()
        out = lm(input_ids=torch.tensor([prefix_ids], device=dev), past_key_values=cache, use_cache=True,
                 logits_to_keep=1)
        first = out.logits[0, -1].float()
        first_lp = torch.log_softmax(first, -1)
        n_ctx = len(prefix_ids)
        stop = torch.tensor(stop_ids, device=dev)
        lp = np.full((len(conts), max(len(c) for c in conts)), np.nan, dtype=np.float32)
        for s in range(0, len(conts), chunk):
            block = conts[s : s + chunk]
            fed = [c[:-1] for c in block]
            starts = np.cumsum([0] + [len(f) for f in fed])
            T = int(starts[-1])
            inp = torch.tensor([t for f in fed for t in f], device=dev)[None]
            pos = torch.tensor([n_ctx + j for f in fed for j in range(len(f))], device=dev)[None]
            mask = torch.zeros(T, n_ctx + T, dtype=torch.bool, device=dev)
            mask[:, :n_ctx] = True
            for a, b in zip(starts[:-1], starts[1:]):
                mask[a:b, n_ctx + a : n_ctx + b] = torch.ones(b - a, b - a, dtype=torch.bool, device=dev).tril()
            branch = copy.deepcopy(cache) if len(conts) > chunk else cache
            logp = torch.log_softmax(
                lm(input_ids=inp, position_ids=pos, attention_mask=mask[None, None], past_key_values=branch,
                   use_cache=True).logits[0].float(), -1)
            # 续写 k 的第 j 个 token (j>=1) 由序列位置 starts[k]+j-1 预测
            tgt_pos, tgt_tok = [], []
            for k, c in enumerate(block):
                for j in range(1, len(c) - 1):
                    tgt_pos.append(starts[k] + j - 1)
                    tgt_tok.append(c[j])
            tok_lp = logp[torch.tensor(tgt_pos, device=dev), torch.tensor(tgt_tok, device=dev)].cpu().numpy()
            end_pos = torch.tensor([starts[k] + len(c) - 2 for k, c in enumerate(block)], device=dev)
            stop_lp = torch.logsumexp(logp[end_pos][:, stop], -1).cpu().numpy()
            c0 = first_lp[torch.tensor([c[0] for c in block], device=dev)].cpu().numpy()
            w = 0
            for k, c in enumerate(block):
                n = len(c)
                lp[s + k, 0] = c0[k]
                lp[s + k, 1 : n - 1] = tok_lp[w : w + n - 2]
                lp[s + k, n - 1] = stop_lp[k]
                w += n - 2
            del branch, logp
    return lp, first


# --------------------------------------------------------------------------
# 运行
# --------------------------------------------------------------------------

NULL_QUERIES = ["", "N/A", "[MASK]"]
VARIANTS = {"sum": ("sum", False), "sum+term": ("sum", True), "avg": ("avg", False), "avg+term": ("avg", True)}


def selfcheck_full_forward(lm, prefix_ids, conts, stop_ids, lp) -> float:
    """每个选项单独整条前向 (前缀 + 续写), 与分块 cache 路径的逐 token log P 比. 返回最大绝对偏差."""
    import torch

    worst = 0.0
    n = len(prefix_ids)
    for k, c in enumerate(conts):
        ids = torch.tensor([prefix_ids + c[:-1]], device=lm.device)
        with torch.no_grad():
            h = lm.model(input_ids=ids).last_hidden_state[0, n - 1 :]
            logp = torch.log_softmax(lm.lm_head(h).float(), -1)
        want = [logp[j, t].item() for j, t in enumerate(c[:-1])]
        want.append(torch.logsumexp(logp[len(c) - 1, stop_ids], -1).item())
        worst = max(worst, float(np.abs(lp[k, : len(c)] - np.array(want)).max()))
    return worst


def main() -> None:
    import argparse
    import json
    import pathlib
    import random
    import time

    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from decidophobia.banking77 import DATA_COMMIT, load_banking77
    from decidophobia.metrics import summarize
    from decidophobia.thermal import ThermalGuard

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--dtype", default="fp32", choices=["bf16", "fp32"],
                    help="bf16 的舍入在 77 路上足以翻转 argmax, 采集默认 fp32")
    ap.add_argument("--limit", type=int, default=0, help="0 = 全量 3080; 否则按 shuffle-seed 打乱后取前 N")
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--perms", type=int, default=3, help="菜单顺序套数; perm0 字母序, 其余随机")
    ap.add_argument("--perm-seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=77, help="一次前向打包几个续写; 77 = 全部一次")
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--examples", type=int, default=6, help="txt 里写几条逐 token 明细")
    ap.add_argument("--temp-max", type=float, default=85.0)
    ap.add_argument("--temp-cooldown", type=float, default=20.0)
    ap.add_argument("--data-dir", default="data/banking77")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    t0 = time.time()
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model)
    lm = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to("cuda").eval()
    guard = ThermalGuard(max_c=args.temp_max, cooldown_s=args.temp_cooldown)

    _, te = load_banking77(args.data_dir)
    K = len(te.names)
    names = [te.names[c] for c in range(K)]
    order_idx = list(range(len(te.queries)))
    if args.limit:
        random.Random(args.shuffle_seed).shuffle(order_idx)
        order_idx = order_idx[: args.limit]
    queries = [te.queries[i] for i in order_idx]
    y = np.array([te.labels[i] for i in order_idx])
    N = len(y)

    # 菜单顺序: orders[p][j] = 第 p 套菜单第 j 行是哪个类
    orders = [sorted(range(K), key=lambda c: names[c])]
    for p in range(1, args.perms):
        o = list(range(K))
        random.Random(args.perm_seed * 1000 + p).shuffle(o)
        orders.append(o)
    conds = [f"perm{p}" for p in range(args.perms)] + ["bare"]

    def prefix_for(cond: str, q: str) -> str:
        if cond == "bare":
            return render_prefix(q, [], with_menu=False)
        return render_prefix(q, [names[c] for c in orders[int(cond[4:])]])

    # 续写按类顺序排, 所有条件共用: 分块的组成在条件之间完全相同, 条件间的差只来自提示
    conts = [tok.encode(" " + n + "\n", add_special_tokens=False) for n in names]
    name_len = np.array([len(c) - 1 for c in conts])
    stop_ids = stop_token_ids(tok)
    L = max(len(c) for c in conts)
    first_ids = sorted({c[0] for c in conts})

    checks: dict = {}
    # 拼接处不得跨界合并: 整条编码必须等于前缀编码 + 续写编码
    checks["boundary_ok"] = all(
        tok.encode(prefix_for(cond, queries[0]) + " " + n + "\n", add_special_tokens=False)
        == tok.encode(prefix_for(cond, queries[0]), add_special_tokens=False) + c
        for cond in conds for n, c in zip(names, conts)
    )
    nl_id = tok.encode("\n", add_special_tokens=False)[0]
    checks["every_continuation_ends_in_single_newline_token"] = all(c[-1] == nl_id for c in conts)

    def score(text: str):
        ids = tok.encode(text, add_special_tokens=False)
        lp, first = score_continuations(lm, ids, conts, stop_ids, args.chunk)
        return ids, lp, first

    ids0, lp0, _ = score(prefix_for("perm0", queries[0]))
    checks["full_forward_max_abs_lp_diff"] = selfcheck_full_forward(lm, ids0, conts, stop_ids, lp0)

    # 空消息先验, 每套菜单顺序各一份
    null_lp = np.stack([
        np.stack([score(prefix_for(f"perm{p}", nq))[1] for nq in NULL_QUERIES]) for p in range(args.perms)
    ])  # (P, 3, K, L)

    LP = {c: np.full((N, K, L), np.nan, dtype=np.float32) for c in conds}
    first_top = {c: np.zeros((N, 10), dtype=np.int32) for c in conds}
    first_top_lp = {c: np.zeros((N, 10), dtype=np.float32) for c in conds}
    first_menu_mass = {c: np.zeros(N, dtype=np.float32) for c in conds}
    prompt_tokens = {c: 0 for c in conds}
    first_ids_t = torch.tensor(first_ids, device="cuda")
    t_loop = time.time()
    print(f"n={N} K={K} conds={conds} chunk={args.chunk} dtype={args.dtype} tctl {guard.read()}", flush=True)
    for i, q in enumerate(queries):
        if i % 50 == 0:
            guard.wait()
        for c in conds:
            ids, lp, first = score(prefix_for(c, q))
            LP[c][i] = lp
            flp = torch.log_softmax(first, -1)
            tv, ti = flp.topk(10)
            first_top[c][i] = ti.cpu().numpy()
            first_top_lp[c][i] = tv.cpu().numpy()
            first_menu_mass[c][i] = torch.logsumexp(flp[first_ids_t], -1).exp().item()
            if i == 0:
                prompt_tokens[c] = len(ids)
        if (i + 1) % 100 == 0 or i + 1 == N:
            el = time.time() - t_loop
            print(f"  {i + 1}/{N}  {el:.0f}s  eta {el / (i + 1) * (N - i - 1):.0f}s  tctl {guard.read()}", flush=True)

    # ---------------------------------------------------------------- 分析
    def scores_of(lp: np.ndarray, v: str) -> np.ndarray:
        if v == "first":
            return lp[..., 0].astype(np.float64)
        how, term = VARIANTS[v]
        return aggregate(lp.astype(np.float64), how, term)

    variants = list(VARIANTS) + ["first"]
    S = {c: {v: scores_of(LP[c], v) for v in variants} for c in conds}
    null_S = {p: {v: scores_of(null_lp[p], v).mean(0) for v in variants} for p in range(args.perms)}
    perm_conds = [f"perm{p}" for p in range(args.perms)]

    def logq(s: np.ndarray) -> np.ndarray:
        return np.log(restricted_softmax(s))

    # 派生条件: 跨顺序平均 (对数空间), PMI 校正 (减去空消息先验), 两者叠加
    for v in variants:
        pmi = {p: S[f"perm{p}"][v] - null_S[p][v][None] for p in range(args.perms)}
        S.setdefault("perm_avg", {})[v] = np.mean([logq(S[c][v]) for c in perm_conds], 0)
        S.setdefault("pmi_perm0", {})[v] = pmi[0]
        S.setdefault("pmi_avg", {})[v] = np.mean([logq(pmi[p]) for p in range(args.perms)], 0)

    def evaluate(s: np.ndarray) -> dict:
        q1 = restricted_softmax(s)
        qh, temps = heldout_temperature(s, y, folds=args.folds)
        yl = y.tolist()
        raw = summarize(q1.tolist(), yl, n_bins=args.n_bins)
        cal = summarize(qh.tolist(), yl, n_bins=args.n_bins)
        pred = q1.argmax(-1)
        rank = (s > s[np.arange(N), y][:, None]).sum(-1)
        return {
            "T1": {**raw, "entropy_bits_mean": float(entropy_bits(q1).mean()),
                   "pred_max_class_frac": float(np.bincount(pred, minlength=K).max() / N)},
            "heldout_T": {**cal, "temps": temps, "entropy_bits_mean": float(entropy_bits(qh).mean())},
            "gold_rank_median": float(np.median(rank)),
            "n_ties_at_top": int(((s == s.max(-1, keepdims=True)).sum(-1) > 1).sum()),
        }

    report_conds = conds + ["perm_avg", "pmi_perm0", "pmi_avg"]
    metrics = {c: {v: evaluate(S[c][v]) for v in variants} for c in report_conds}

    stability: dict = {}
    for v in variants:
        qs = [restricted_softmax(S[c][v]) for c in perm_conds]
        stability[v] = {
            "across_perms_T1": pairwise_agreement(qs),
            "menu_perm0_vs_bare_argmax_agree": float(
                (S["perm0"][v].argmax(-1) == S["bare"][v].argmax(-1)).mean()),
        }
    stability["across_variants_perm0_argmax_agree"] = {
        f"{a}|{b}": float((S["perm0"][a].argmax(-1) == S["perm0"][b].argmax(-1)).mean())
        for i_, a in enumerate(variants) for b in variants[i_ + 1 :]
    }

    bias: dict = {"position_mass": {}, "length_profile": {}}
    for v in ["sum", "avg"]:
        for p in range(args.perms):
            pm = position_mass(restricted_softmax(S[f"perm{p}"][v]), np.broadcast_to(np.array(orders[p]), (N, K)))
            bias["position_mass"][f"{v}/perm{p}"] = {
                "first5": pm[:5].round(4).tolist(), "last5": pm[-5:].round(4).tolist(),
                "max_over_uniform": float(pm.max() * K), "min_over_uniform": float(pm.min() * K),
                "argmax_line": int(pm.argmax()), "all": pm.round(5).tolist(),
            }
        bias["length_profile"][v] = length_profile(restricted_softmax(S["perm_avg"][v]), y, name_len)

    collected: dict = {}
    for c in conds:
        m = np.exp(S[c]["sum+term"]).sum(-1)  # 77 个完整答案 (含终止) 的总概率, 事件互斥
        top1 = first_top[c][:, 0]
        uniq, cnt = np.unique(top1, return_counts=True)
        top_hist = [[tok.decode([int(uniq[j])]), int(cnt[j])] for j in np.argsort(-cnt)[:8]]
        collected[c] = {
            "menu_string_mass_mean": float(m.mean()),
            "menu_string_mass_p05_p50_p95": np.percentile(m, [5, 50, 95]).round(5).tolist(),
            "first_step_menu_token_mass_mean": float(first_menu_mass[c].mean()),
            "first_step_top1_is_menu_first_token_rate": float(np.isin(top1, first_ids).mean()),
            "first_step_top1_histogram": top_hist,
            "prompt_tokens_item0": prompt_tokens[c],
        }
    collected["name_token_len_hist"] = {int(k): int(v) for k, v in zip(*np.unique(name_len, return_counts=True))}
    collected["n_distinct_first_tokens"] = len(first_ids)
    collected["null_prior"] = {
        v: {"top3": [[names[int(k)], float(restricted_softmax(null_S[0][v][None])[0, k])]
                     for k in np.argsort(-null_S[0][v])[:3]],
            "entropy_bits": float(entropy_bits(restricted_softmax(null_S[0][v][None]))[0])}
        for v in ["sum", "avg"]
    }

    # ---------------------------------------------------------------- 逐 token 明细
    lines = []
    for i in range(min(args.examples, N)):
        s = S["perm0"]["sum"][i]
        qv = restricted_softmax(s[None])[0]
        top = np.argsort(-s)[:5]
        lines.append(f"[{i}] {queries[i]!r}\n    gold: {names[y[i]]}  (rank {int((s > s[y[i]]).sum())}, q={qv[y[i]]:.4f})")
        for k in top:
            lines.append(f"    q={qv[k]:.4f}  sum={s[k]:8.3f}  avg={S['perm0']['avg'][i][k]:7.3f}  {names[k]}")
        for tag, k in [("gold", int(y[i])), ("argmax", int(top[0]))]:
            toks = [tok.decode([t]) for t in conts[k]]
            parts = "  ".join(f"{t!r}:{v:.2f}" for t, v in zip(toks, LP["perm0"][i, k, : len(conts[k])]))
            lines.append(f"    {tag:6s} tokens  {parts}")
        ft = "  ".join(f"{tok.decode([int(t)])!r}:{np.exp(v):.3f}" for t, v in zip(first_top["perm0"][i][:6], first_top_lp["perm0"][i][:6]))
        lines.append(f"    'Answer:' 之后模型自己想说的前 6 个 token: {ft}\n")

    meta = {
        "model_id": args.model,
        "model_revision": pathlib.Path(snapshot_download(args.model, local_files_only=True)).name,
        "dtype": args.dtype,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "dataset": "PolyAI-LDN/task-specific-datasets banking_data test",
        "dataset_commit": DATA_COMMIT,
        "n_items": N, "n_choices": K, "limit": args.limit, "shuffle_seed": args.shuffle_seed,
        "perms": args.perms, "perm_seed": args.perm_seed, "chunk": args.chunk, "folds": args.folds,
        "null_queries": NULL_QUERIES, "stop_tokens": [tok.decode([t]) for t in stop_ids],
        "question": QUESTION,
        "prompt_example_perm0": prefix_for("perm0", queries[0]),
        "prompt_example_bare": prefix_for("bare", queries[0]),
        "elapsed_sec": round(time.time() - t0, 1),
        "loop_sec_per_item": round((time.time() - t_loop) / N, 3),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
    }
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{time.strftime('%Y%m%d-%H%M%S')}-choice-loglik-banking77-{args.model.replace('/', '_')}-{args.dtype}-n{N}"
    np.savez_compressed(
        out / (stem + ".npz"), labels=y, item_index=np.array(order_idx), orders=np.array(orders),
        name_len=name_len, null_lp=null_lp, **{f"lp_{c}": LP[c] for c in conds},
        **{f"first_top_{c}": first_top[c] for c in conds}, **{f"first_top_lp_{c}": first_top_lp[c] for c in conds},
    )
    payload = {"meta": meta, "selfchecks": checks, "collected": collected, "metrics": metrics,
               "stability": stability, "bias": bias, "names": names}
    (out / (stem + ".json")).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    (out / (stem + ".txt")).write_text("\n".join(lines))

    print("\n".join(lines))
    print(json.dumps({"selfchecks": checks, "collected": collected}, indent=1, ensure_ascii=False))
    print(f"{'cond':10s} {'variant':9s} {'acc':>6s} {'top5':>6s} {'nll':>6s} {'ece':>6s} {'conf':>6s} {'H':>5s}"
          f"  | T_heldout {'nll':>6s} {'ece':>6s}  maxfrac")
    for c in report_conds:
        for v in variants:
            r, h = metrics[c][v]["T1"], metrics[c][v]["heldout_T"]
            print(f"{c:10s} {v:9s} {r['accuracy']:6.3f} {r['top5_accuracy']:6.3f} {r['nll']:6.3f} {r['ece']:6.3f} "
                  f"{r['conf_mean']:6.3f} {r['entropy_bits_mean']:5.2f}  | {np.mean(h['temps']):6.2f}    "
                  f"{h['nll']:6.3f} {h['ece']:6.3f}  {r['pred_max_class_frac']:.3f}")
    print(json.dumps(stability, indent=1))
    print(json.dumps({k: {kk: {x: vv[x] for x in vv if x != 'all'} for kk, vv in b.items()} if k == "position_mass" else b
                      for k, b in bias.items()}, indent=1))
    print(f"\n→ {out / (stem + '.json')}")


if __name__ == "__main__":
    main()
