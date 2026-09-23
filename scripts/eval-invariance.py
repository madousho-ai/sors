#!/usr/bin/env python
"""不变性诊断: 同一道题只改一个变量, 看训好的决策头选中的描述变不变. 只评估, 不训练.

题目是训练中评估的 synth256 (512 个合成意图的 1024 条消息, 每题 256 项菜单, 连续编号), 与 --eval-synth 同一批.
变体 (都和原菜单比, 按描述对齐):
  repeat        原样再打一次分, 只换 batch 组合 (padding 不同) —— 噪声底
  renumbered_a/_b  行打乱, 仍连续编号 —— 部署形态, 行和码一起变
  rows_only     行打乱, 每条描述带着原来的码 —— 只动行
  codes_only    行不动, 256 个码重新随机分配 —— 只动码
  short_random  正确描述 + 59 个随机描述, 按原顺序, 连续编号 —— 只砍长度
  short_top     正确描述 + 模型在原菜单上打分最高的 59 个错误描述 —— 只留模型自己最容易混的
rows_only / codes_only 必然是不连续编号 (连续编号下行号就是码号, 分不开); 只在这里用, 训练中的评估仍连续编号.

每个变体报: 准确率、metrics.consistency (flip_rate / tv_mean / gold_logp_drift)、全词表格式读数 (m_answer / top1_in).

  PYTHONPATH=src .venv/bin/python scripts/eval-invariance.py --init runs/<run>/trained.pt [--model Qwen/Qwen3-1.7B-Base]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from decidophobia.data import random_rows, reassigned_codes, reorder_menu, shuffled_rows, top_rows
from decidophobia.metrics import consistency, summarize
from decidophobia.model import prepare_model
from decidophobia.prompt import DEFAULT_LAYOUT, LAYOUTS
from decidophobia.synth import synth_eval_examples
from decidophobia.tokens import install_d_tokens, install_type_tokens
from decidophobia.train import load_trained, score_examples

SHORT = 60


def variants(base, seed: int) -> dict[str, list]:
    """不需要模型打分的那些变体. 每个变体一个独立的 rng, 加减一个变体不改变其它变体."""
    def rng(i):
        return random.Random(seed * 100 + i)

    r = {k: rng(i) for i, k in enumerate(("a", "b", "rows", "codes", "short"))}
    return {
        "repeat": list(base),
        "renumbered_a": [shuffled_rows(e, r["a"], keep_codes=False) for e in base],
        "renumbered_b": [shuffled_rows(e, r["b"], keep_codes=False) for e in base],
        "rows_only": [shuffled_rows(e, r["rows"], keep_codes=True) for e in base],
        "codes_only": [reassigned_codes(e, r["codes"]) for e in base],
        "short_random": [reorder_menu(e, random_rows(e, SHORT, r["short"])) for e in base],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="trained.pt")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base", help="必须与训练时的基模相同; trained.pt 里没记")
    ap.add_argument("--layout", default=DEFAULT_LAYOUT, choices=LAYOUTS)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=4, help="256 项菜单的 batch; 60 项的用它的 4 倍")
    ap.add_argument("--limit", type=int, default=0, help="只用前几道题 (0 = 全部 1024), 冒烟用")
    ap.add_argument("--seed", type=int, default=0, help="题目与 train.py --seed 相同时是同一批 synth256; 也定变体的 rng")
    ap.add_argument("--out", default=None, help="默认 results/invariance-<run 目录名>.json")
    args = ap.parse_args()

    base = synth_eval_examples(256, args.seed + 256)
    if args.limit:
        base = base[: args.limit]
    tok = AutoTokenizer.from_pretrained(args.model)
    d_ids = install_d_tokens(tok)
    train_ids = d_ids + install_type_tokens(tok)
    lm = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")
    m = prepare_model(lm, train_ids, lora_r=8, lora_alpha=16, lora_dropout=0.0)
    cfg = load_trained(m, train_ids, args.init)

    def score(exs, shuffle_seed=None):
        """shuffle_seed 给了就打乱题目顺序再打分、再按原顺序放回 —— 每道题所在 batch 的邻居不同."""
        order = list(range(len(exs)))
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(order)
        k = max(len(e.options) for e in exs)
        bs = args.batch_size if k > SHORT else args.batch_size * 4
        s = score_examples(m, tok, d_ids, [exs[i] for i in order], bs, k, args.max_length, args.layout)
        back = [0] * len(order)
        for j, i in enumerate(order):
            back[i] = j
        return {key: [v[back[i]] for i in range(len(order))] for key, v in s.items()}

    def report(name, exs, s, t0):
        n = len(exs)
        rec = {"variant": name, "k": max(len(e.options) for e in exs), **summarize(s["q"], s["gold"]),
               "m_answer_mean": sum(s["m_answer"]) / n, "top1_in_menu_rate": sum(s["top1_in"]) / n,
               "t": round(time.time() - t0, 1)}
        if name != "base":
            rec.update(consistency(base_s["q"], base, s["q"], exs))
            rec["picks"] = [e.options[max(range(len(e.options)), key=q.__getitem__)] for e, q in zip(exs, s["q"])]
        flip = f"flip {rec['flip_rate']:.3f}  tv {rec['tv_mean']:.3f}  drift {rec['gold_logp_drift']:.3f}  " if name != "base" else ""
        print(f"{name:13s} k={rec['k']:3d}  acc {rec['accuracy']:.3f}  {flip}m_answer {rec['m_answer_mean']:.3f}  "
              f"top1_in {rec['top1_in_menu_rate']:.3f}  {rec['t']:.0f}s", flush=True)
        return rec

    print(f"init {args.init}  model {args.model}  questions {len(base)}", flush=True)
    t0 = time.time()
    base_s = score(base)
    records = [report("base", base, base_s, t0)]
    runs = variants(base, args.seed)
    top = [reorder_menu(e, top_rows(q[: len(e.options)], e.gold_idx, SHORT)) for e, q in zip(base, base_s["q"])]
    for name, exs in list(runs.items()) + [("short_top", top)]:
        t0 = time.time()
        s = score(exs, shuffle_seed=args.seed + 1) if name == "repeat" else score(exs)
        records.append(report(name, exs, s, t0))

    out = pathlib.Path(args.out or f"results/invariance-{pathlib.Path(args.init).parent.name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "train_config": cfg, "gold": [e.label for e in base],
                               "base_picks": [e.options[max(range(len(e.options)), key=q.__getitem__)]
                                              for e, q in zip(base, base_s["q"])],
                               "records": records}, indent=1))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
