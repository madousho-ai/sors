#!/usr/bin/env python
"""训练 256 槽决策头 (LoRA + D-token 嵌入).

--dataset 是用加号连起来的数据集列表, 一个 batch 里各占一份:
  banking77   Banking77, 按类留出 17 个测泛化 (seen / unseen)
  boolq       BoolQ, k=2, 逐条问句
  synth       datasets/synth-intents, 512 个合成意图 (不在训练里时可用 --eval-synth 当留出评估)
  massive     MASSIVE 的 train 分区, 60 个语音助手意图; test 分区留给 scripts/eval-massive.py
banking77 与 synth 同时在时, 两者的类并进一个 id 空间, 菜单干扰项从并集里抽 —— 这就是 k 能拉到 256 的来源.
massive 的上下文标签不同, 自成一池.
"both" 仍可用, 等于 banking77+boolq.

  PYTHONPATH=src .venv/bin/python scripts/train.py --dataset banking77 --steps 2000
  PYTHONPATH=src .venv/bin/python scripts/train.py --dataset boolq --init runs/<b77>/trained.pt --steps 0   # 跨任务零训练评估
  PYTHONPATH=src .venv/bin/python scripts/train.py --dataset banking77+boolq+synth --k-max 256 --k-log --grad-ckpt --steps 2000
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

from decidophobia.data import RandomCodes, class_split, menu_k_range, merge_sets
from decidophobia.loss import LOSSES
from decidophobia.model import LORA_TARGETS, prepare_model
from decidophobia.prompt import DEFAULT_LAYOUT, LAYOUTS
from decidophobia.thermal import ThermalGuard
from decidophobia.tokens import install_d_tokens, install_type_tokens
from decidophobia.train import EvalSet, TrainConfig, load_trained, save_trained, train

KNOWN = ("banking77", "boolq", "synth", "massive")


def parse_datasets(spec: str) -> list[str]:
    names = ["banking77", "boolq"] if spec == "both" else spec.split("+")
    bad = [n for n in names if n not in KNOWN]
    if bad or len(set(names)) != len(names):
        raise SystemExit(f"--dataset: unknown or repeated {bad or names}; use + to combine {KNOWN}")
    return names


def build_data(args):
    """每个数据集给一个 sampler 和若干评估集. 返回 (sample_fn, eval_sets, split_info).
    只读数据, 不碰模型 —— tests/test_train_cli.py 直接调它."""
    datasets = parse_datasets(args.dataset)
    if args.eval_synth and "synth" in datasets:
        raise SystemExit("--eval-synth: synth is in training, so it would not be a held-out evaluation")
    if not 0.0 <= args.random_codes <= 1.0:
        raise SystemExit(f"--random-codes is a share of training menus, 0..1; got {args.random_codes}")
    erng = random.Random(args.seed + 1)
    samplers, eval_sets, split_info = [], {}, {}
    ktr = menu_k_range(args.k_min, args.k_max)
    # banking77 与 synth 共用一个类 id 空间: 菜单干扰项从两边的并集里抽
    b77 = synth = None
    if "banking77" in datasets:
        from decidophobia.banking77 import load_banking77

        b77 = load_banking77(args.data_dir)
    if "synth" in datasets:
        from decidophobia.synth import load_synth

        synth, synth_domains = load_synth()
    if b77 and synth:
        tr, offs = merge_sets(b77[0], synth)
        te = merge_sets(b77[1], synth)[0]  # synth 的 test 就是它自己的 1024 条, 只用来撑类 id 空间, 评估不抽它
        synth_classes = list(range(offs[1], offs[1] + len(synth.names)))
    elif b77:
        tr, te = b77
        synth_classes = []
    elif synth:
        tr = te = synth
        synth_classes = list(range(len(synth.names)))
    if b77:
        split = class_split(len(b77[0].names), args.held_out, seed=args.seed)
        kr = (args.k_eval, args.k_eval)
        # 评估菜单只从 Banking77 自己的类里抽, 与历史 run 可比
        eval_sets["seen"] = EvalSet(te.build_examples(split.train, kr, erng), args.eval_batch_size)
        eval_sets["unseen"] = EvalSet(te.build_examples(split.held_out, kr, erng), args.eval_batch_size)
        if synth_classes:
            # 加一档: 留出类的题, 菜单用合成意图填到 k_eval (默认 256), 每个槽都当得上正确答案.
            # 提示约 2600 token, batch 缩到 1/4: 评估时前向会建 KV cache, 16 条要 ~5 GiB
            ex_far = te.build_examples(split.held_out, kr, erng, pool=split.held_out + synth_classes)
            eval_sets[f"unseen{len(ex_far[0].options)}"] = EvalSet(ex_far, max(1, args.eval_batch_size // 4))
        pool = split.train + synth_classes
        samplers.append(lambda n, rng: tr.sample_examples(split.train, ktr, n, rng, pool=pool, k_log=args.k_log))
        split_info = {"train": split.train, "held_out": split.held_out,
                      "held_out_names": [b77[0].names[c] for c in split.held_out]}
    if synth:
        s_pool = (split.train if b77 else []) + synth_classes
        samplers.append(lambda n, rng: tr.sample_examples(synth_classes, ktr, n, rng, pool=s_pool, k_log=args.k_log))
        split_info["synth_classes"] = len(synth_classes)
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
        eval_sets["boolq"] = EvalSet(bva.build_examples([0, 1], (2, 2), erng), max(1, args.eval_batch_size // 2), pos_class=1)
        samplers.append(lambda n, rng: btr.sample_examples([0, 1], (2, 2), n, rng))
    if args.eval_synth:
        # 留出评估: 512 个合成意图的 1024 条消息, 菜单只含合成意图. 60 项那档的正确答案都在 D0..D59,
        # 256 项那档散到 D0..D255 —— 训练菜单不到 256 时, 看没当过答案的码能不能用.
        from decidophobia.synth import synth_eval_examples

        eval_sets["synth60"] = EvalSet(synth_eval_examples(60, args.seed + 60), args.eval_batch_size)
        eval_sets["synth256"] = EvalSet(synth_eval_examples(256, args.seed + 256), max(1, args.eval_batch_size // 4))
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
    ap.add_argument("--dataset", default="banking77", help="banking77 / boolq / synth 用 + 连接; both = banking77+boolq")
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
    ap.add_argument("--k-log", action="store_true", help="配 --k-min: 长度按对数均匀取, 默认均匀")
    ap.add_argument("--k-eval", type=int, default=256,
                    help="评估菜单最多几项, 池子不够就全放: seen 60, unseen 17, 带 synth 时留出类 + 合成意图 256; BoolQ 恒为 2")
    ap.add_argument("--held-out", type=int, default=17, help="Banking77 留出的类数, 训练里完全不出现")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--eval-limit", type=int, default=0, help="每个评估集最多用几条 (0 = 全部)")
    ap.add_argument("--eval-synth", action="store_true",
                    help="把 synth 当留出评估集: synth60 / synth256 两档, 只含合成意图. 与 --dataset 里的 synth 互斥")
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
