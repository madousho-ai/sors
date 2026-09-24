#!/usr/bin/env python
"""在 MASSIVE (en-US test, 2974 条, 60 intent) 上评估训好的决策头. 只评估, 不训练.

  PYTHONPATH=src .venv/bin/python scripts/eval-massive.py \
      --init runs/<banking77>/trained.pt --init runs/<boolq>/trained.pt --init runs/<both>/trained.pt

不给 --init 时评估的是「未训练」: D 行随机初始化、LoRA 为零, 这是 chance 参照.
给了 --init 时第一个记录仍是未训练 (--no-untrained 关掉).

每个 init × k 报 acc / nll / ece / conf_mean. k 的三档:
  10  与训练分布对齐 (gold + 9 干扰), 直接对标 Banking77 unseen
  20  训练里没见过的菜单长度, 测格式外推
  60  全菜单, 每个 intent 列一次, 论文报 MASSIVE 时的形态 (fine-tuned XLM-R en-US 约 0.88)
产出 results/massive.json: {"records": [{"init", "tag", "k", "chance", <metrics>}...]}.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from decidophobia.massive import load_massive
from decidophobia.model import prepare_model
from decidophobia.prompt import DEFAULT_LAYOUT, LAYOUTS
from decidophobia.thermal import ThermalGuard
from decidophobia.tokens import install_d_tokens, install_type_tokens
from decidophobia.train import EvalSet, evaluate, prepare_from_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="append", default=[], help="trained.pt, 可多个; 每个评一遍")
    ap.add_argument("--no-untrained", action="store_true", help="不评未训练基线")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--k", type=int, nargs="+", default=[10, 20, 60])
    ap.add_argument("--layout", default=DEFAULT_LAYOUT, choices=LAYOUTS)
    ap.add_argument("--type-marker", action="store_true")
    ap.add_argument("--max-length", type=int, default=1024, help="k=60 的菜单约 500 token, 512 会从左截掉指令")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="最多评几条 (0 = 全部 2974)")
    ap.add_argument("--seed", type=int, default=0, help="菜单组法与未训练 D 行初始化的种子")
    ap.add_argument("--temp-max", type=float, default=85.0)
    ap.add_argument("--temp-cooldown", type=float, default=20.0)
    ap.add_argument("--data-dir", default="data/massive")
    ap.add_argument("--out", default="results/massive.json")
    args = ap.parse_args()

    te = load_massive(args.data_dir)
    classes = list(range(len(te.names)))
    eval_sets = {}
    for k in args.k:
        exs = te.build_examples(classes, (k, k), random.Random(args.seed + k))
        if args.limit:
            random.Random(args.seed).shuffle(exs)
            exs = exs[: args.limit]
        eval_sets[k] = EvalSet(exs, args.batch_size)

    tok = AutoTokenizer.from_pretrained(args.model)
    d_ids = install_d_tokens(tok)
    train_ids = d_ids + install_type_tokens(tok)
    guard = ThermalGuard(max_c=args.temp_max, cooldown_s=args.temp_cooldown)

    def build(path):
        """每份档各搭一次模型: LoRA 形状照档里记的, 几份档的 rank 可以各不相同. 返回 (模型, 档里的训练 config).
        种子每次都重置, 未训练基线与档里没有的类型行在每次搭建时都是同一份初值."""
        torch.manual_seed(args.seed)
        lm = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")
        if path is None:  # 未训练: LoRA B 为零, rank 不影响输出
            return prepare_model(lm, train_ids, lora_r=8, lora_alpha=16, lora_dropout=0.0), None
        return prepare_from_checkpoint(lm, train_ids, path)

    inits: list[tuple[str, str | None]] = []
    if not args.no_untrained:
        inits.append(("untrained", None))
    inits += [(pathlib.Path(p).parent.name, p) for p in args.init]

    records = []
    print(f"massive test n={len(te.queries)} classes={len(classes)}  k={args.k}  inits={len(inits)}  tctl {guard.read()}", flush=True)
    for tag, path in inits:
        m, cfg = build(path)
        for k, es in eval_sets.items():
            guard.wait()
            t0 = time.time()
            r = evaluate(m, tok, d_ids, es, k_max=k, max_length=args.max_length,
                         layout=args.layout, type_marker=args.type_marker)
            rec = {"init": path, "tag": tag, "k": k, "chance": 1 / k, "train_config": cfg, **r, "t": round(time.time() - t0, 1)}
            records.append(rec)
            print(f"{tag:52s} k={k:2d}  acc {r['accuracy']:.4f}  nll {r['nll']:.3f}  ece {r['ece']:.3f}  "
                  f"conf {r['conf_mean']:.3f}  n {r['n']}  {rec['t']:.0f}s  tctl {guard.read()}", flush=True)
            print("    by gold slot  " + "  ".join(f"{b}: {s['accuracy']:.3f} (n {s['n']})"
                                                   for b, s in r["by_gold_slot"].items()), flush=True)
        del m
        torch.cuda.empty_cache()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "names": te.names, "records": records}, indent=2))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
