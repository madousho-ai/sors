"""训练循环与评估. 每一步的菜单都是现组的: 同一条 query 每次见到的选项集和位置都不同.

train() 不认识数据集: 拿一个 sample_fn (给 n 和 rng, 还 n 条 MenuExample) 和若干 EvalSet.
单数据集、双数据集混合、只评估不训练 (steps=0), 都是调用方组 sample_fn 的事.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch

from decidophobia.batch import collate
from decidophobia.data import MenuExample
from decidophobia.loss import answer_mass, gather_slot_logits, training_loss
from decidophobia.metrics import answer_mass_summary, binary_summary, by_gold_slot, menu_size_summary, summarize
from decidophobia.model import last_logits, trainable_param_groups
from decidophobia.prompt import DEFAULT_LAYOUT
from decidophobia.schedule import lr_scale

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
    loss: str = "all-slots"  # loss.LOSSES: all-slots 分母是全部 D 槽; menu 只有菜单 k 个槽
    eval_every: int = 100
    log_every: int = 20
    seed: int = 0


@dataclass
class EvalSet:
    examples: list[MenuExample]
    batch_size: int = 16
    pos_class: int | None = None  # 二元数据集给正类 id, 就多报 auroc / pos_rate / brier_binary


@torch.no_grad()
def evaluate(m, tok, d_ids, es: EvalSet, k_max: int, max_length: int, layout: str, type_marker: bool = False) -> dict:
    """summarize() 那组指标 (位置空间), 二元集再加 binary_summary (类空间). 概率只在各自菜单的 k 个槽上归一.
    另报格式遵从 (answer_mass_summary): 全词表下有多少概率落在菜单的槽上, 与 baseline 脚本的 m_answer 同一个量."""
    was_training = m.training
    m.eval()
    dev = next(m.parameters()).device
    Q, Y, MA, OFF, TOP = [], [], [], [], []
    for s in range(0, len(es.examples), es.batch_size):
        chunk = es.examples[s : s + es.batch_size]
        b = collate(chunk, tok, d_ids, k_max, layout, max_length, type_marker)
        b = {k: v.to(dev) for k, v in b.items()}
        logits = last_logits(m, b["input_ids"], b["attention_mask"])
        q = torch.softmax(gather_slot_logits(logits, b["slot_ids"]), dim=-1)  # pad 槽 exp(-inf)=0
        Q.extend(q.cpu().tolist())
        Y.extend(b["gold"].tolist())
        ma, off, top1 = answer_mass(logits, b["slot_ids"], d_ids)
        MA.extend(ma.cpu().tolist())
        OFF.extend(off.cpu().tolist())
        TOP.extend(top1.cpu().tolist())
    if was_training:
        m.train()
    out = summarize(Q, Y)
    if es.pos_class is not None:
        out.update(binary_summary(Q, es.examples, es.pos_class))
    out.update(answer_mass_summary(MA, OFF, TOP))
    out.update(menu_size_summary([len(ex.options) for ex in es.examples]))
    out["n"] = len(Y)
    out["by_gold_slot"] = by_gold_slot(Q, Y)
    return out


def scalar_items(prefix: str, d: dict) -> list[tuple[str, float]]:
    """把 evaluate() 的结果拍平成 TensorBoard 标量: 嵌套 dict 接成 a/b/c, None 丢掉."""
    out = []
    for k, v in d.items():
        if isinstance(v, dict):
            out += scalar_items(f"{prefix}/{k}", v)
        elif v is not None:
            out.append((f"{prefix}/{k}", v))
    return out


def train(
    m, tok, d_ids: list[int], sample_fn: SampleFn, eval_sets: dict[str, EvalSet],
    cfg: TrainConfig, log_path=None, writer=None, guard=None,
) -> list[dict]:
    """跑 cfg.steps 步 (0 = 只做 step 0 的评估). 返回评估记录. 每条记录也追加写到 log_path.

    writer: torch.utils.tensorboard.SummaryWriter, 可选. 标量分三组:
      train/loss, train/lr_*        每 log_every 步
      eval/<set>/<metric>           每次评估
      sys/tctl_c, sys/thermal_waits 温度与被温度闸拦下的次数
    guard: ThermalGuard, 可选. 每步之前和每次评估之前各问一次.
    """
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    history: list[dict] = []
    log_f = open(log_path, "a") if log_path else None
    waits = 0

    def tctl() -> float | None:
        return guard.read() if guard else None

    def do_eval(step: int, train_loss: float | None):
        nonlocal waits
        if guard:
            waits += guard.wait()
        rec = {"step": step, "train_loss": train_loss, "t": round(time.time() - t0, 1),
               "tctl_c": tctl(), "thermal_waits": waits}
        for name, es in eval_sets.items():
            rec[name] = evaluate(m, tok, d_ids, es, cfg.k_max, cfg.max_length, cfg.layout, cfg.type_marker)
            if writer:
                for tag, v in scalar_items(f"eval/{name}", rec[name]):
                    writer.add_scalar(tag, v, step)
        if writer:
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
    do_eval(0, None)
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
    running = 0.0
    for step in range(1, cfg.steps + 1):
        if guard:
            waits += guard.wait()
        exs = sample_fn(cfg.batch_size, rng)
        b = collate(exs, tok, d_ids, cfg.k_max, cfg.layout, cfg.max_length, cfg.type_marker)
        b = {k: v.to("cuda") for k, v in b.items()}
        logits = last_logits(m, b["input_ids"], b["attention_mask"])
        loss = training_loss(cfg.loss, logits, b["slot_ids"], b["gold"], d_ids)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        running += loss.item()
        if step % cfg.log_every == 0:
            avg = running / cfg.log_every
            t = tctl()
            print(f"step {step:5d}  loss {avg:.4f}  {time.time() - t0:.0f}s"
                  + (f"  tctl {t:.0f}°C" if t is not None else ""), flush=True)
            if writer:
                writer.add_scalar("train/loss", avg, step)
                for i, g in enumerate(opt.param_groups):
                    writer.add_scalar(f"train/lr_group{i}", g["lr"], step)
                if t is not None:
                    writer.add_scalar("sys/tctl_c", t, step)
            running = 0.0
        if step % cfg.eval_every == 0 or step == cfg.steps:
            do_eval(step, loss.item())
    if log_f:
        log_f.close()
    return history


def save_trained(m, train_ids: list[int], cfg: TrainConfig, path) -> None:
    """只存会变的部分: LoRA 权重 + 放开的嵌入行 (D 行 + 类型行) + 配置. 基模照 model_id 重新加载."""
    rows = m.get_input_embeddings().rows
    state = {n: p.detach().cpu() for n, p in m.named_parameters() if p.requires_grad and "lora_" in n}
    torch.save(
        {"lora": state, "d_embed": rows.detach().cpu(), "d_ids": train_ids, "config": asdict(cfg)},
        path,
    )


def load_trained(m, train_ids: list[int], path) -> dict:
    """把 save_trained 存的 LoRA 权重和嵌入行灌回 prepare_model 之后的模型. 返回存档里的 config.

    档里的 ids 允许是模型 train_ids 的前缀: 类型 token 加进来之前的档只有 256 个 D 行,
    那 3 行当时不在提示里、梯度为零, 留在初始化就是那次训练的真实状态. 多出的行原样不动.
    """
    ck = torch.load(path, map_location="cpu")
    n = len(ck["d_ids"])
    if ck["d_ids"] != train_ids[:n]:
        raise ValueError(f"checkpoint's {n} trainable embedding rows are not a prefix of this model's {len(train_ids)}")
    params = dict(m.named_parameters())
    missing = [n for n in ck["lora"] if n not in params]
    if missing:
        raise ValueError(f"{len(missing)} LoRA tensors in checkpoint have no home in this model, e.g. {missing[0]}")
    with torch.no_grad():
        for name, t in ck["lora"].items():
            params[name].copy_(t.to(params[name].dtype))
        rows = m.get_input_embeddings().rows
        rows[:n].copy_(ck["d_embed"].to(rows.dtype).to(rows.device))
    return ck["config"]
