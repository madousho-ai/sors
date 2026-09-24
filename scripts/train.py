#!/usr/bin/env python
"""训练 256 槽决策头 (LoRA + D-token 嵌入).

--dataset 是用加号连起来的训练集列表, 一个 batch 里各占一份:
  banking77   Banking77, 按类留出 17 个测泛化 (seen / unseen)
  boolq       BoolQ, k=2, 逐条问句
  synth       datasets/synth-intents, 4096 个合成意图 × 3 条消息. 每条消息两道题, 各占一份:
              菜单题只列正确意图所在领域的意图 (k 256 即整个领域 256 个), 二元题问消息里的一个细节 (no / yes)
  massive     MASSIVE 的 train 分区, 60 个语音助手意图
每个训练集各自组菜单, 干扰项不跨集合抽.
"both" 仍可用, 等于 banking77+boolq.

--eval 是评估集列表, 与训练集无关, 默认 banking77+massive+boolq+simple (见 build_eval_sets).

  PYTHONPATH=src .venv/bin/python scripts/train.py --dataset synth --grad-ckpt --steps 2000
  PYTHONPATH=src .venv/bin/python scripts/train.py --init runs/<run>/trained.pt --steps 0   # 只评估
  .venv/bin/tensorboard --logdir runs

产出 (--out 目录):
  log.jsonl     每次评估一行 (step 0 是训练前 / 加载后的基线)
  result.json   配置 + 类切分 + 全部评估记录
  trained.pt    LoRA 权重 + 256 个 D 行嵌入 (--steps 0 时不写, 保住 --init 那份)
  tb/           TensorBoard 事件
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer

from decidophobia.data import RandomCodes, class_split, menu_k_range
from decidophobia.loss import LOSSES
from decidophobia.model import LORA_TARGETS, prepare_model
from decidophobia.prompt import DEFAULT_LAYOUT, LAYOUTS
from decidophobia.thermal import ThermalGuard
from decidophobia.tokens import install_d_tokens, install_type_tokens
from decidophobia.train import EvalSet, TrainConfig, load_trained, save_trained, train

KNOWN = ("banking77", "boolq", "synth", "massive")
KNOWN_EVAL = ("banking77", "massive", "boolq", "simple")


def _parse_list(spec: str, known: tuple[str, ...], flag: str) -> list[str]:
    names = spec.split("+")
    bad = [n for n in names if n not in known]
    if bad or len(set(names)) != len(names):
        raise SystemExit(f"{flag}: unknown or repeated {bad or names}; use + to combine {known}")
    return names


def parse_datasets(spec: str) -> list[str]:
    return _parse_list("banking77+boolq" if spec == "both" else spec, KNOWN, "--dataset")


def build_eval_sets(args, b77_test=None, boolq_val=None) -> dict[str, EvalSet]:
    """--eval 列的评估集, 与训练集无关. 菜单全量、连续编号; 每个集合的菜单用 Random(seed + 菜单长度) 组,
    MASSIVE 那份因此与 scripts/eval-massive.py 的 k=60 逐题相同. 训练里已读过的 split 可以传进来复用.
    simple 是 datasets/synth-simple-eval, 菜单写死在文件里 (不受 --k-eval / --seed 影响), 展开成每个菜单长度一个集合."""
    out = {}
    for name in _parse_list(args.eval, KNOWN_EVAL, "--eval"):
        if name == "simple":
            from decidophobia.simple_eval import load_simple_eval

            for sub, exs in load_simple_eval().items():
                pos = 1 if sub == "simple_bool" else None
                out[sub] = EvalSet(exs, args.eval_batch_size, pos_class=pos)
            continue
        if name == "banking77":
            if b77_test is None:
                from decidophobia.banking77 import load_banking77

                b77_test = load_banking77(args.data_dir)[1]
            te, bs, pos = b77_test, args.eval_batch_size, None
        elif name == "massive":
            from decidophobia.massive import load_massive

            te, bs, pos = load_massive(partition="test"), args.eval_batch_size, None
        else:
            if boolq_val is None:
                from decidophobia.boolq import load_boolq

                boolq_val = load_boolq()[1]
            te, bs, pos = boolq_val, max(1, args.eval_batch_size // 2), 1
        classes = list(range(len(te.names)))
        k = min(args.k_eval, len(classes))
        out[name] = EvalSet(te.build_examples(classes, (k, k), random.Random(args.seed + k)), bs, pos_class=pos)
    return out


def build_data(args):
    """每个数据集给一个 sampler 和若干评估集. 返回 (sample_fn, eval_sets, split_info).
    只读数据, 不碰模型 —— tests/test_train_cli.py 直接调它."""
    datasets = parse_datasets(args.dataset)
    if not 0.0 <= args.random_codes <= 1.0:
        raise SystemExit(f"--random-codes is a share of training menus, 0..1; got {args.random_codes}")
    erng = random.Random(args.seed + 1)
    samplers, eval_sets, split_info = [], {}, {}
    ktr = menu_k_range(args.k_min, args.k_max)
    b77 = None
    if "banking77" in datasets:
        from decidophobia.banking77 import load_banking77

        b77 = load_banking77(args.data_dir)
        tr, te = b77
        split = class_split(len(tr.names), args.held_out, seed=args.seed)
        kr = (args.k_eval, args.k_eval)
        eval_sets["seen"] = EvalSet(te.build_examples(split.train, kr, erng), args.eval_batch_size)
        eval_sets["unseen"] = EvalSet(te.build_examples(split.held_out, kr, erng), args.eval_batch_size)
        samplers.append(lambda n, rng: tr.sample_examples(split.train, ktr, n, rng, k_log=args.k_log))
        split_info = {"train": split.train, "held_out": split.held_out,
                      "held_out_names": [tr.names[c] for c in split.held_out]}
    if "synth" in datasets:
        # 每条消息两道题: 菜单题只列正确意图所在领域的意图 (k 256 即整个领域), 二元题问消息里的一个细节.
        # 两种题各占一个 sampler, 于是一批里各一半.
        from decidophobia.synth import load_synth, load_synth_binary, sample_domain_menus

        synth, synth_domains = load_synth()
        synth_bin = load_synth_binary()
        samplers.append(lambda n, rng: sample_domain_menus(synth, synth_domains, ktr, n, rng))
        samplers.append(lambda n, rng: synth_bin.sample_examples([0, 1], (2, 2), n, rng))
        split_info["synth_classes"] = len(synth.names)
    if "massive" in datasets:
        # MASSIVE 的 train 分区 (11514 条, 60 意图). 上下文标签是 Voice command, 不与 banking77 并池:
        # 各自全量菜单 60 项. 它的 test 分区留给 scripts/eval-massive.py.
        from decidophobia.massive import load_massive

        mtr = load_massive(partition="train")
        m_classes = list(range(len(mtr.names)))
        samplers.append(lambda n, rng: mtr.sample_examples(m_classes, ktr, n, rng))
        split_info["massive_classes"] = len(m_classes)
    if "boolq" in datasets:
        from decidophobia.boolq import load_boolq

        btr, bva = load_boolq()
        samplers.append(lambda n, rng: btr.sample_examples([0, 1], (2, 2), n, rng))
    eval_sets.update(build_eval_sets(args, b77[1] if b77 else None, bva if "boolq" in datasets else None))
    if args.eval_limit:
        for k, es in eval_sets.items():
            exs = list(es.examples)
            random.Random(args.seed + 2).shuffle(exs)
            eval_sets[k] = EvalSet(exs[: args.eval_limit], es.batch_size, es.pos_class)

    random_codes = RandomCodes(args.random_codes)  # 一个 run 一个: 它记着每个码上过几次菜单

    def sample_fn(n, rng):
        """一批里各数据集平分 (第一个 sampler 拿零头), 再打乱. --random-codes 只作用在这里:
        评估集始终按位置编号 D0, D1, ..., 与部署时调用方写的菜单同形."""
        parts = [n // len(samplers)] * len(samplers)
        parts[0] += n - sum(parts)
        out = [ex for s, c in zip(samplers, parts) for ex in s(c, rng)]
        rng.shuffle(out)
        return random_codes(out, rng)

    return sample_fn, eval_sets, split_info


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synth",
                    help="训练集, banking77 / boolq / synth / massive 用 + 连接; both = banking77+boolq")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--init", default=None, help="从这份 trained.pt 加载 LoRA + D 行再开始 (或配 --steps 0 只评估)")
    ap.add_argument("--trainable", default="attn", choices=sorted(LORA_TARGETS),
                    help="放开的范围: d-only 只训 D 行; attn 加 attention LoRA; attn-mlp 再加 MLP LoRA")
    ap.add_argument("--layout", default=DEFAULT_LAYOUT, choices=LAYOUTS,
                    help="context-first: 上下文在前, 前缀可作 KV cache 共享 (默认); menu-first: 菜单在前, 对照组")
    ap.add_argument("--type-marker", action="store_true",
                    help="问句标签写成 'Question (<|bool|>):', 类型 token 随 D 行一起训")
    ap.add_argument("--loss", default="vocab", choices=LOSSES,
                    help="vocab: 分母是整个词表, 普通 token 每步被压低, 答题位置只说 D 码; "
                         "all-slots: 分母是全部 256 个 D 槽; menu: 只在菜单 k 个槽上归一 (后两种是旧版)")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--lr-embed", type=float, default=1e-3)
    ap.add_argument("--lr-schedule", default="cosine", choices=["constant", "cosine"])
    ap.add_argument("--warmup", type=int, default=100, help="线性 warmup 步数")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=300, help="0 = 只评估")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=4096,
                    help="超长提示从左截. 实测最长: 256 项菜单 2685, BoolQ 1277, Banking77 全 60 项 444")
    ap.add_argument("--grad-ckpt", action="store_true", help="梯度 checkpointing: 激活 5 GiB -> 0.6 GiB, 时间 +30%%")
    ap.add_argument("--k-min", type=int, default=None,
                    help="不给 = 全量菜单: 池子里的选项全放进去, 最多 --k-max 项. 给了才在 k-min..k-max 随机抽长度 (旧行为)")
    ap.add_argument("--k-max", type=int, default=256, help="菜单最多几项 (D 槽只有 256 个)")
    ap.add_argument("--k-log", action="store_true", help="配 --k-min: 长度按对数均匀取, 默认均匀. 只作用于 banking77")
    ap.add_argument("--k-eval", type=int, default=256,
                    help="评估菜单最多几项, 池子不够就全放: seen 60, unseen 17, --eval 的 banking77 77 / massive 60; BoolQ 恒为 2")
    ap.add_argument("--held-out", type=int, default=17, help="Banking77 留出的类数, 训练里完全不出现")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--eval-limit", type=int, default=0, help="每个评估集最多用几条 (0 = 全部)")
    ap.add_argument("--eval", default="banking77+massive+boolq+simple",
                    help="评估集, 用 + 连接, 与 --dataset 无关: banking77 (test 3080 条, 77 类全量菜单) / "
                         "massive (test 2974 条, 60 类全量菜单) / boolq (validation 3270 条) / "
                         "simple (synth-simple-eval: 消息直接说出答案, 5..255 项各 10 题 + 10 道 no/yes)")
    ap.add_argument("--random-codes", type=float, default=0.0,
                    help="选择题里换成随机码的比例 (0..1): 挑上菜单次数最少的 k 个 D 码、顺序随机, 每个码上菜单时是答案的概率都是 1/k, "
                         "整场下来 D0..D255 当答案的次数期望相同 (60 项菜单下 rate >= 0.77 才补得齐). BoolQ 永远 D0 / D1. "
                         "0 = 全部按位置 D0, D1, ... (旧行为). 评估集不受影响")
    ap.add_argument("--temp-max", type=float, default=85.0, help="CPU Tctl 超过就暂停 (°C)")
    ap.add_argument("--temp-cooldown", type=float, default=20.0, help="每次暂停多少秒")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default="data/banking77")
    ap.add_argument("--out", default=None, help="默认 runs/<时间戳>-<dataset>-<trainable>-<schedule>-<layout>")
    return ap


def run_tag(args) -> str:
    """run 目录名的后缀. 旧 loss 的写法保持不变 (all-slots -> -allslots, menu 不加), 旧 run 的名字照旧能复现."""
    return ("-qtype" if args.type_marker else "") + (f"-b{args.batch_size}" if args.batch_size != 8 else "") \
        + (f"-kfull{args.k_max}" if args.k_min is None else f"-k{args.k_min}-{args.k_max}") \
        + ("-klog" if args.k_log else "") \
        + (f"-rcodes{args.random_codes:g}" if args.random_codes > 0 else "") \
        + {"all-slots": "-allslots", "vocab": "-vocab"}.get(args.loss, "")


def main() -> None:
    args = build_parser().parse_args()
    datasets = parse_datasets(args.dataset)

    tag = run_tag(args)
    out = pathlib.Path(args.out or f"runs/{time.strftime('%Y%m%d-%H%M%S')}-{args.dataset}-{args.trainable}-{args.lr_schedule}-{args.layout}{tag}")
    out.mkdir(parents=True, exist_ok=True)
    sample_fn, eval_sets, split_info = build_data(args)

    # ---- 模型 ---------------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(args.model)
    d_ids = install_d_tokens(tok)
    t_ids = install_type_tokens(tok)
    train_ids = d_ids + t_ids  # 类型行永远放开; 不带 --type-marker 时它们不出现在提示里, 梯度为零、原地不动
    lm = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")
    m = prepare_model(lm, train_ids, args.lora_r, args.lora_alpha, args.lora_dropout,
                      trainable=args.trainable, grad_ckpt=args.grad_ckpt)
    init_cfg = load_trained(m, train_ids, args.init) if args.init else None

    k_pad = max([args.k_max, args.k_eval] + [len(e.options) for es in eval_sets.values() for e in es.examples])
    cfg = TrainConfig(
        steps=args.steps, batch_size=args.batch_size, k_max=k_pad, max_length=args.max_length,
        lr_lora=args.lr_lora, lr_embed=args.lr_embed, weight_decay=args.weight_decay,
        lr_schedule=args.lr_schedule, warmup_steps=args.warmup, layout=args.layout, type_marker=args.type_marker,
        loss=args.loss, eval_every=args.eval_every, seed=args.seed,
    )
    guard = ThermalGuard(max_c=args.temp_max, cooldown_s=args.temp_cooldown)
    writer = SummaryWriter(log_dir=str(out / "tb"))
    writer.add_text("args", json.dumps(vars(args), indent=2), 0)
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"dataset={'+'.join(datasets)} trainable={args.trainable} layout={args.layout} loss={args.loss} "
          f"random_codes={args.random_codes:g} "
          f"k={menu_k_range(args.k_min, args.k_max)}{' log' if args.k_log else ''} params {n_train:,}  "
          f"init={args.init or '-'}  eval " + " ".join(f"{k}={len(v.examples)}" for k, v in eval_sets.items())
          + f"  tctl {guard.read()}  → {out}", flush=True)
    history = train(m, tok, d_ids, sample_fn, eval_sets, cfg, log_path=out / "log.jsonl", writer=writer, guard=guard)
    writer.close()
    if args.steps > 0:
        save_trained(m, train_ids, cfg, out / "trained.pt")
    (out / "result.json").write_text(json.dumps({
        "args": vars(args),
        "trainable_params": n_train,
        "init_config": init_cfg,
        "split": split_info,
        "history": history,
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
    }, indent=2))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
