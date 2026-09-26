"""训练循环. 每一步的菜单都是现组的: 同一条 query 每次见到的选项集和位置都不同.

train() 不认识数据集: 拿一个 sample_fn (给 n 和 rng, 还 n 条 MenuExample) 和若干 EvalSet.
评估点上的打分与指标在 decidophobia.scoring, 这里只决定什么时候评、评哪些.
单数据集、双数据集混合、只评估不训练 (steps=0), 都是调用方组 sample_fn 的事.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from decidophobia.batch import collate, pair_alignment
from decidophobia.data import MenuExample, arrangements
from decidophobia.loss import consistency_js, menu_hits, smooth_target, training_loss
from decidophobia.model import grouped_last_logits, trainable_param_groups
from decidophobia.prompt import DEFAULT_LAYOUT
from decidophobia.schedule import lr_scale
from decidophobia.scoring import EvalSet, consistency_eval, evaluate

SampleFn = Callable[[int, random.Random], list[MenuExample]]


@dataclass
class TrainConfig:
    steps: int = 300
    batch_size: int = 8
    k_max: int = 16
    max_length: int = 512
    lr_lora: float = 1e-4
    lr_embed: float = 1e-3
    weight_decay: float = 0.0
    lr_schedule: str = "constant"  # constant | cosine
    warmup_steps: int = 0
    layout: str = DEFAULT_LAYOUT  # context-first | menu-first
    type_marker: bool = False  # 'Question (<|bool|>):' 里带类型 token
    loss: str = "all-slots"  # loss.LOSSES: vocab 整个词表; all-slots 全部 D 槽; menu 菜单 k 个槽. scripts/train.py 总是显式传, 默认 vocab
    label_smoothing: float = 0.0  # 目标分布里摊到菜单各行的份额, 见 loss.smooth_target; 0 = 不平滑
    consistency: float = 0.0  # 一致性项的权重 λ, 见 step_loss; > 0 时 sample_fn 要给成对的题 (data.with_partners)
    eval_every: int = 100
    save_every: int = 0  # 每隔几步交一次存档给 train() 的 on_checkpoint, 最后一步除外 (调用方另存); 0 = 途中不存
    probe_size: int = 200  # 训练中的评估点每个评估集抽几道题, 见 probe_passes
    probe_passes: int = 5  # 每道题排成几种随机的样子 (行打乱、码随机), 训练中与最后一步都用这个数
    micro_batches: int = 1  # 每步的 prompt 按长度分几组各自前向 (model.grouped_last_logits); 1 = 整批一次, 旧行为
    log_every: int = 20
    seed: int = 0


def scalar_items(prefix: str, d: dict) -> list[tuple[str, float]]:
    """把 evaluate() 的结果拍平成 TensorBoard 标量: 嵌套 dict 接成 a/b/c, None 丢掉."""
    out = []
    for k, v in d.items():
        if isinstance(v, dict):
            out += scalar_items(f"{prefix}/{k}", v)
        elif v is not None:
            out.append((f"{prefix}/{k}", v))
    return out


def probe_passes(examples: list[MenuExample], size: int, passes: int, key: str) -> list[list[MenuExample]]:
    """训练中每个评估点都用的那一套: 抽 size 道 (不够就全部, 顺序照旧), 排成 passes 种随机的样子
    (data.arrangements: 行打乱、码随机). 全由 key 定死, 整场训练每个评估点比的都是同一批题、同样的排法,
    曲线上两点的差别只来自模型. 用自己的 rng, 不动训练抽题的那一个."""
    rng = random.Random(key)
    if len(examples) > size:
        examples = [examples[i] for i in sorted(rng.sample(range(len(examples)), size))]
    return arrangements(examples, passes, rng)


def eval_record(m, tok, d_ids, eval_sets: dict[str, EvalSet], probes: dict[str, list[list[MenuExample]]],
                cfg: TrainConfig, final: bool) -> dict:
    """一个评估点的读数, 按评估集名分:
      consistency       探针子集 (probe_passes) 上的一致性. 每个评估点都有
      final=True 时再加两样, 都用全量评估集:
      eval              部署形态 (连续编号) 下的 evaluate(): 正确率、NLL、校准等
      consistency_full  cfg.probe_passes 种随机排法下的一致性, 排法由 seed 和集名定死"""
    args = (cfg.k_max, cfg.max_length, cfg.layout, cfg.type_marker)
    rec = {"consistency": {name: consistency_eval(m, tok, d_ids, probes[name], es.batch_size, *args)
                           for name, es in eval_sets.items()}}
    if final:
        rec["eval"] = {name: evaluate(m, tok, d_ids, es, *args) for name, es in eval_sets.items()}
        rec["consistency_full"] = {
            name: consistency_eval(m, tok, d_ids, arrangements(es.examples, cfg.probe_passes,
                                                               random.Random(f"full-{cfg.seed}-{name}")),
                                   es.batch_size, *args)
            for name, es in eval_sets.items()}
    return rec


def step_target(exs: list[MenuExample], b: dict, eps: float) -> torch.Tensor | None:
    """一步训练用的目标分布. 全是硬标签且不平滑时返回 None, 损失照旧只认 gold (旧 run 逐位复现);
    否则是 collate 给的 target (硬标签那几行是 one-hot), 再按 eps 在各自菜单上平滑."""
    if eps == 0 and all(ex.target is None for ex in exs):
        return None
    return smooth_target(b["target"], b["slot_ids"], eps)


def step_loss(
    cfg: TrainConfig, exs: list[MenuExample], b: dict, logits: torch.Tensor, d_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """一步的 (总损失, 交叉熵, JS). 交叉熵是 training_loss, 对 batch 里每一条取平均.
    cfg.consistency > 0 时 exs 必须是 data.with_partners 排好的 [a, a', b, b', ...], 总损失再加
    consistency · consistency_js (两种排法按描述对齐后的 JS, 按对平均); 不成对就报 ValueError.
    consistency 0 时 JS 是 None、总损失就是交叉熵那个张量."""
    ce = training_loss(cfg.loss, logits, b["slot_ids"], b["gold"], d_ids,
                       target=step_target(exs, b, cfg.label_smoothing))
    if cfg.consistency <= 0:
        return ce, ce, None
    align = pair_alignment(exs, b["slot_ids"].shape[1]).to(logits.device)
    js = consistency_js(logits, b["slot_ids"], align)
    return ce + cfg.consistency * js, ce, js


def train(
    m, tok, d_ids: list[int], sample_fn: SampleFn, eval_sets: dict[str, EvalSet],
    cfg: TrainConfig, log_path=None, writer=None, guard=None, on_checkpoint: Callable[[int], None] | None = None,
) -> list[dict]:
    """跑 cfg.steps 步 (0 = 只评估). 返回评估记录. 每条记录也追加写到 log_path.

    评估点是 step 0 与每 eval_every 步, 以及最后一步. 每个评估点都跑探针 (eval_record 的 consistency):
    每个评估集固定抽 cfg.probe_size 道, 固定 cfg.probe_passes 种随机排法. 最后一步 (steps 0 时就是 step 0)
    再跑全量的 eval 与 consistency_full.

    on_checkpoint(step): cfg.save_every > 0 时, 每 save_every 步做完 (同一步有评估就在评估之后) 叫一次,
    由调用方把 m 存下来. 最后一步不叫, 训练结束后调用方本来就存.

    writer: torch.utils.tensorboard.SummaryWriter, 可选. 标量:
      train/loss, train/lr_*        每 log_every 步. train/loss 只是交叉熵, 与加一致性项之前的 run 同一个量
      train/accuracy                每 log_every 步, 这几步训练批上的正确率 (loss.menu_hits: 菜单上 logit 最高的一行
                                    是不是目标分布的最大行; 均匀标签的题不计). 批里是打乱过、换过码的题, 成对时两份都算
      train/js                      cfg.consistency > 0 时, 成对题的 JS (乘 λ 之前)
      consistency/<set>/<metric>    每个评估点, 探针子集上的 accuracy / agree / js
      eval/<set>/<metric>           最后一步, 全量评估集的正确率那一套 (与旧 run 的同名标量同一个量)
      consistency_full/<set>/<metric>  最后一步, 全量评估集的一致性
      sys/tctl_c, sys/thermal_waits 温度与被温度闸拦下的次数
    guard: ThermalGuard, 可选. 每步之前和每次评估之前各问一次.

    log_path 的每条记录另有 train_accuracy / train_accuracy_n: 上一个评估点以来全部训练题上的正确率与题数
    (step 0 还没训练, 是 None / 0).
    """
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    history: list[dict] = []
    log_f = open(log_path, "a") if log_path else None
    waits = 0
    since_eval = [0, 0]  # 上一个评估点以来训练题的 (答对, 计入)
    probes = {name: probe_passes(es.examples, cfg.probe_size, cfg.probe_passes, f"probe-{cfg.seed}-{name}")
              for name, es in eval_sets.items()}

    def tctl() -> float | None:
        return guard.read() if guard else None

    def do_eval(step: int, train_loss: float | None, final: bool):
        nonlocal waits
        if guard:
            waits += guard.wait()
        hits, n = since_eval
        since_eval[:] = [0, 0]
        rec = {"step": step, "train_loss": train_loss, "train_accuracy": hits / n if n else None,
               "train_accuracy_n": n, "t": round(time.time() - t0, 1), "tctl_c": tctl(), "thermal_waits": waits}
        rec.update(eval_record(m, tok, d_ids, eval_sets, probes, cfg, final))
        if writer:
            for group in ("consistency", "eval", "consistency_full"):
                for name, r in rec.get(group, {}).items():
                    for tag, v in scalar_items(f"{group}/{name}", r):
                        writer.add_scalar(tag, v, step)
            if rec["tctl_c"] is not None:
                writer.add_scalar("sys/tctl_c", rec["tctl_c"], step)
            writer.add_scalar("sys/thermal_waits", waits, step)
            writer.flush()
        history.append(rec)
        line = json.dumps(rec)
        print(line, flush=True)
        if log_f:
            log_f.write(line + "\n")
            log_f.flush()

    t0 = time.time()
    m.train()
    do_eval(0, None, final=cfg.steps == 0)
    if cfg.steps == 0:
        if log_f:
            log_f.close()
        return history

    opt = torch.optim.AdamW(
        trainable_param_groups(m, cfg.lr_lora, cfg.lr_embed), weight_decay=cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_scale(s, cfg.warmup_steps, cfg.steps, cfg.lr_schedule)
    )
    running, running_js, running_hits, running_n = 0.0, 0.0, 0, 0
    dev = next(m.parameters()).device
    for step in range(1, cfg.steps + 1):
        if guard:
            waits += guard.wait()
        exs = sample_fn(cfg.batch_size, rng)
        b = collate(exs, tok, d_ids, cfg.k_max, cfg.layout, cfg.max_length, cfg.type_marker)
        b = {k: v.to(dev) for k, v in b.items()}
        logits = grouped_last_logits(m, b["input_ids"], b["attention_mask"], cfg.micro_batches)
        loss, ce, js = step_loss(cfg, exs, b, logits, d_ids)
        hits, n = menu_hits(logits.detach(), b["slot_ids"], b["target"])  # 更新之前的模型在这一批上的读数
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        running += ce.item()
        running_hits, running_n = running_hits + hits, running_n + n
        since_eval[0] += hits
        since_eval[1] += n
        if js is not None:
            running_js += js.item()
        if step % cfg.log_every == 0:
            avg = running / cfg.log_every
            avg_js = running_js / cfg.log_every
            acc = running_hits / running_n if running_n else None
            t = tctl()
            print(f"step {step:5d}  loss {avg:.4f}" + (f"  js {avg_js:.4f}" if js is not None else "")
                  + (f"  acc {acc:.3f}" if acc is not None else "")
                  + f"  {time.time() - t0:.0f}s" + (f"  tctl {t:.0f}°C" if t is not None else ""), flush=True)
            if writer:
                writer.add_scalar("train/loss", avg, step)
                if acc is not None:
                    writer.add_scalar("train/accuracy", acc, step)
                if js is not None:
                    writer.add_scalar("train/js", avg_js, step)
                for i, g in enumerate(opt.param_groups):
                    writer.add_scalar(f"train/lr_group{i}", g["lr"], step)
                if t is not None:
                    writer.add_scalar("sys/tctl_c", t, step)
            running, running_js, running_hits, running_n = 0.0, 0.0, 0, 0
        if step % cfg.eval_every == 0 or step == cfg.steps:
            do_eval(step, ce.item(), final=step == cfg.steps)
        if on_checkpoint and cfg.save_every > 0 and step % cfg.save_every == 0 and step < cfg.steps:
            on_checkpoint(step)
    if log_f:
        log_f.close()
    return history
