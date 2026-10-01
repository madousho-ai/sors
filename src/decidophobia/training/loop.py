"""训练循环. 每一步的菜单都是现组的: 同一条 query 每次见到的选项集和位置都不同.

train() 不认识数据集: 拿一个 sample_fn (给 n 和 rng, 还 n 条 MenuExample) 和若干 EvalSet.
评估点上的打分与指标在 decidophobia.evaluation.scoring, 这里只决定什么时候评、评哪些.
单数据集、双数据集混合、只评估不训练 (steps=0), 都是调用方组 sample_fn 的事.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch

from decidophobia.core.batch import collate, pair_alignment, trim_left_padding
from decidophobia.core.menu import MenuExample, arrangements
from decidophobia.core.model import decision_logits, grouped_last_logits, last_logits, select_batch, trainable_param_groups
from decidophobia.core.decision import architecture_config
from decidophobia.core.prompt import DEFAULT_LAYOUT
from decidophobia.evaluation.scoring import EvalSet, consistency_eval, evaluate
from decidophobia.training.loss import consistency_js, menu_hits, smooth_target, training_loss
from decidophobia.training.schedule import lr_scale

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
    context_marker: bool = False  # state 两头包 <|context_start|> <|context_end|>, 超长只截包裹里的 state
    loss: str = "all-slots"  # loss.LOSSES: vocab 整个词表; all-slots 全部 D 槽; menu 菜单 k 个槽. scripts/train.py 总是显式传, 默认 vocab
    label_smoothing: float = 0.0  # 目标分布里摊到菜单各行的份额, 见 loss.smooth_target; 0 = 不平滑
    consistency: float = 0.0  # 一致性项的权重 λ, 见 step_loss; > 0 时 sample_fn 要给成对的题 (menu.with_partners)
    eval_every: int = 100
    save_every: int = 0  # 每隔几步交一次存档给 train() 的 on_checkpoint, 最后一步除外 (调用方另存); 0 = 途中不存
    probe_size: int = 200  # 训练中的评估点每个评估集抽几道题, 见 probe_passes
    probe_passes: int = 5  # 每道题排成几种随机的样子 (行打乱、码随机), 训练中与最后一步都用这个数
    micro_batches: int = 1  # 每步的 prompt 按长度分几组各自前向 (model.grouped_last_logits); 1 = 整批一次, 旧行为
    log_every: int = 20
    seed: int = 0
    accumulate_gradients: bool = False  # 每组立即反传，保持一个逻辑 batch 一次更新
    candidate_prefix_cache: str = "off"  # CLI defaults new candidates to auto; old configs retain full forwards.


class Fp32Master:
    """低精度 (bf16 / fp16) 的可训参数各配一份 fp32 主权重, 优化器只更新主权重.

    bf16 在 1.0 附近的间距是 2^-8, 单步 1e-5 量级的改动直接写回 bf16 会被舍入掉, 权重原地不动;
    在 fp32 上累积, 攒够一个间距 bf16 那份才跟着变. 全参时的 bf16 主干要这个.
    fp32 的参数原样留在 params 里 (peft 的 LoRA 权重本来就是 fp32), 旧 run 逐位不变.

    每步: backward 之后 pull_grads() 把梯度搬到主权重上 (模型上那份清掉, 否则下一次 backward 累加上去),
    opt.step() 之后 push() 把主权重写回模型."""

    def __init__(self, params: list[torch.Tensor]):
        self.pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.params: list[torch.Tensor] = []
        for p in params:
            if p.dtype in (torch.bfloat16, torch.float16):
                w = p.detach().float().clone().requires_grad_(True)
                self.pairs.append((p, w))
                self.params.append(w)
            else:
                self.params.append(p)

    def pull_grads(self) -> None:
        for p, w in self.pairs:
            w.grad = None if p.grad is None else p.grad.float()
            p.grad = None

    @torch.no_grad()
    def push(self) -> None:
        for p, w in self.pairs:
            p.copy_(w)

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
    (menu.arrangements: 行打乱、码随机). 全由 key 定死, 整场训练每个评估点比的都是同一批题、同样的排法,
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
    args = (cfg.k_max, cfg.max_length, cfg.layout, cfg.type_marker, cfg.context_marker)
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
    cfg.consistency > 0 时 exs 必须是 menu.with_partners 排好的 [a, a', b, b', ...], 总损失再加
    consistency · consistency_js (两种排法按描述对齐后的 JS, 按对平均); 不成对就报 ValueError.
    consistency 0 时 JS 是 None、总损失就是交叉熵那个张量."""
    ce = training_loss(cfg.loss, logits, b["slot_ids"], b["gold"], d_ids,
                       target=step_target(exs, b, cfg.label_smoothing))
    if cfg.consistency <= 0:
        return ce, ce, None
    align = pair_alignment(exs, b["slot_ids"].shape[1]).to(logits.device)
    js = consistency_js(logits, b["slot_ids"], align)
    return ce + cfg.consistency * js, ce, js


def backward_groups(m, cfg: TrainConfig, exs: list[MenuExample], b: dict, d_ids: list[int]):
    """完整 JS 配对一起分组，每组立即反传；调用方整步只清一次梯度、更新一次。

    CE 按 prompt、JS 按 pair 取平均，完整配对使二者均可乘本组 prompt 数 / 整批 prompt 数。
    返回整步的 detached CE、JS、hits、n，各组的计算图在循环内释放。
    """
    size = len(exs)
    paired = cfg.consistency > 0
    width = 2 if paired else 1
    if size == 0 or size % width or cfg.micro_batches < 1:
        raise ValueError("empty batch, incomplete consistency pairs, or invalid micro_batches")
    alignment = pair_alignment(exs, b["slot_ids"].shape[1]).to(b["input_ids"].device) if paired else None
    target = step_target(exs, b, cfg.label_smoothing)
    lengths = b["attention_mask"].sum(1).reshape(-1, width).max(1).values
    order = torch.argsort(lengths, descending=True, stable=True)
    ce_sum, js_sum, hits_sum, n_sum = 0.0, 0.0, 0, 0
    for units in torch.tensor_split(order, min(cfg.micro_batches, len(order))):
        rows = (units[:, None] * width + torch.arange(width, device=units.device)).flatten()
        if getattr(m, "decision_config", None) is not None:
            logits = decision_logits(m, select_batch(b, rows))
        else:
            input_ids, mask = trim_left_padding(b["input_ids"][rows], b["attention_mask"][rows])
            logits = last_logits(m, input_ids, mask)
        slots, gold, targets = b["slot_ids"][rows], b["gold"][rows], b["target"][rows]
        ce = training_loss(cfg.loss, logits, slots, gold, d_ids,
                           target=None if target is None else target[rows])
        js = consistency_js(logits, slots, alignment[units]) if paired else None
        weight = len(rows) / size
        loss = ce if js is None else ce + cfg.consistency * js
        (weight * loss).backward()
        ce_sum += weight * ce.item()
        if js is not None:
            js_sum += weight * js.item()
        hits, n = menu_hits(logits.detach(), slots, targets)
        hits_sum, n_sum = hits_sum + hits, n_sum + n
        del logits, loss, ce, js
    return ce_sum, js_sum if paired else None, hits_sum, n_sum


def train(
    m, tok, d_ids: list[int], sample_fn: SampleFn, eval_sets: dict[str, EvalSet],
    cfg: TrainConfig, log_path=None, writer=None, guard=None, on_checkpoint: Callable[[int], None] | None = None,
    *, resume: dict | None = None, on_state: Callable[[dict], None] | None = None, stop_after: int | None = None,
) -> list[dict]:
    """跑 cfg.steps 步 (0 = 只评估). 返回评估记录. 每条记录也追加写到 log_path.

    评估点是 step 0 与每 eval_every 步, 以及最后一步. 每个评估点都跑探针 (eval_record 的 consistency):
    每个评估集固定抽 cfg.probe_size 道, 固定 cfg.probe_passes 种随机排法. 最后一步 (steps 0 时就是 step 0)
    再跑全量的 eval 与 consistency_full.

    on_checkpoint(step): cfg.save_every > 0 时, 每 save_every 步做完 (同一步有评估就在评估之后) 叫一次,
    由调用方把 m 存下来. 最后一步不叫, 训练结束后调用方本来就存.

    resume: 完整状态或旧权重恢复的 {step, history}；sample_fn 必须是尚未推进的原数据管线。
    cfg.steps 始终是原目标步数。stop_after 只限制本次执行的更新次数，完整更新后可由 on_state 存档。
    on_state 在 save_every 边界及本次末步收到模型、master、optimizer、RNG 和统计窗口；应同步复制/写盘。
    原采样器内部状态通过完整重放 step 次采样调用重建，native 状态额外校验采样 RNG 一致。

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
    # sample_fn 必须是新建的原版本管线。重建全部采样调用，包含 queue、码计数、问法与配对的随机流。
    architecture = architecture_config(m)
    if architecture["kind"] == "candidate":
        m.set_candidate_prefix_cache(cfg.candidate_prefix_cache)
    if resume and resume.get("architecture", {"kind": "slots"}) != architecture:
        raise ValueError("resume architecture differs from the saved decision model")
    if architecture["kind"] != "slots" and cfg.loss != "menu":
        raise ValueError("decision architectures define a menu distribution; use loss='menu'")
    start = int(resume["step"]) if resume else 0
    if not 0 <= start <= cfg.steps or (stop_after is not None and stop_after < 1):
        raise ValueError("invalid completed step or stop_after")
    if resume and "config" in resume:
        changed = {k for k, v in resume["config"].items() if k not in ("micro_batches", "accumulate_gradients")
                   and asdict(cfg).get(k) != v}
        if changed:
            raise ValueError(f"resume config changed: {sorted(changed)}")
    end = cfg.steps if stop_after is None else min(cfg.steps, start + stop_after)
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    for i in range(start):
        sample_fn(cfg.batch_size, rng)
        if (i + 1) % 250 == 0:
            print(f"resume: restored sampling through step {i + 1}/{start}", flush=True)
    if resume and "rng" in resume and rng.getstate() != resume["rng"]:
        raise ValueError("sampler replay diverged: verify original data, code, arguments and Python version")
    history: list[dict] = list(resume.get("history", [])) if resume else []
    log_f = open(log_path, "a") if log_path else None
    waits = resume.get("waits", 0) if resume else 0
    since_eval = list(resume.get("since_eval", [0, 0])) if resume else [0, 0]
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

    elapsed = resume.get("elapsed", history[-1].get("t", 0) if history else 0) if resume else 0
    t0 = time.time() - elapsed
    m.train()
    if resume is None:
        do_eval(0, None, final=cfg.steps == 0)
    if cfg.steps == 0:
        if log_f:
            log_f.close()
        return history

    params = {n: p for n, p in m.named_parameters() if p.requires_grad}
    full = resume is not None and "optimizer" in resume
    if full:
        if params.keys() != resume["model"].keys():
            raise ValueError("trainable parameter names differ from the training state")
        with torch.no_grad():
            for name, p in params.items():
                if p.shape != resume["model"][name].shape or p.dtype != resume["model"][name].dtype:
                    raise ValueError(f"resume parameter shape or dtype mismatch: {name}")
                p.copy_(resume["model"][name])
    groups = trainable_param_groups(m, cfg.lr_lora, cfg.lr_embed)
    master = Fp32Master(groups[0]["params"])  # 主干那一组 (LoRA 或 full 的主干); rows 那一组照旧
    groups[0]["params"] = master.params
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_scale(s, cfg.warmup_steps, cfg.steps, cfg.lr_schedule)
    )
    if full:
        if len(master.params) != len(resume["master"]):
            raise ValueError("master parameter count differs from the training state")
        with torch.no_grad():
            for p, saved in zip(master.params, resume["master"]):
                if p.shape != saved.shape or p.dtype != saved.dtype:
                    raise ValueError("master parameter shape or dtype mismatch")
                p.copy_(saved)
        opt.load_state_dict(resume["optimizer"])
        sched.load_state_dict(resume["scheduler"])
        torch.set_rng_state(resume["torch_rng"].cpu())
        if resume.get("cuda_rng"):
            torch.cuda.set_rng_state_all([s.cpu() for s in resume["cuda_rng"]])
        random.setstate(resume["python_rng"])
    elif start:
        # 原逻辑第1501次更新用第1500位置的LR；保持总日程，跳过新warmup。
        lrs = [base * lr_scale(start, cfg.warmup_steps, cfg.steps, cfg.lr_schedule) for base in sched.base_lrs]
        for group, lr in zip(opt.param_groups, lrs):
            group["lr"] = lr
        sched.last_epoch, sched._step_count, sched._last_lr = start, start + 1, lrs
    resume_info = dict(resume.get("resume_info", {})) if resume else {}
    if resume and not full:
        resume_info.update(optimizer_reset_at=start, torch_rng_reset_at=start,
                           lr_at_resume=[g["lr"] for g in opt.param_groups])
    running, running_js, running_hits, running_n = resume.get("running", [0.0, 0.0, 0, 0]) if resume else (0.0, 0.0, 0, 0)

    def capture_state(step):
        # callback 必须在返回前复制/写盘；这些张量直接引用当前训练状态。
        return {"step": step, "config": asdict(cfg), "architecture": architecture,
                "model": {n: p.detach() for n, p in params.items()},
                "master": [p.detach() for p in master.params], "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(), "rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if next(m.parameters()).is_cuda else [],
                "python_rng": random.getstate(), "history": list(history), "since_eval": list(since_eval),
                "running": [running, running_js, running_hits, running_n], "waits": waits,
                "elapsed": time.time() - t0, "resume_info": resume_info}

    dev = next(m.parameters()).device
    for step in range(start + 1, end + 1):
        if guard:
            waits += guard.wait()
        exs = sample_fn(cfg.batch_size, rng)
        b = collate(exs, tok, d_ids, cfg.k_max, cfg.layout, cfg.max_length, cfg.type_marker, cfg.context_marker,
                    architecture=architecture["kind"])
        b = {k: v.to(dev) for k, v in b.items()}
        opt.zero_grad(set_to_none=True)
        if cfg.accumulate_gradients:
            ce_value, js_value, hits, n = backward_groups(m, cfg, exs, b, d_ids)
        else:
            logits = (decision_logits(m, b, cfg.micro_batches) if architecture["kind"] != "slots" else
                      grouped_last_logits(m, b["input_ids"], b["attention_mask"], cfg.micro_batches))
            loss, ce, js = step_loss(cfg, exs, b, logits, d_ids)
            hits, n = menu_hits(logits.detach(), b["slot_ids"], b["target"])
            loss.backward()
            ce_value, js_value = ce.item(), js.item() if js is not None else None
            del logits, loss, ce, js
        master.pull_grads()
        opt.step()
        master.push()
        sched.step()
        if cfg.accumulate_gradients:
            opt.zero_grad(set_to_none=True)  # 更新完成后的梯度可立即释放，评估和存档也留出显存
        running += ce_value
        running_hits, running_n = running_hits + hits, running_n + n
        since_eval[0] += hits
        since_eval[1] += n
        if js_value is not None:
            running_js += js_value
        if step % cfg.log_every == 0:
            avg = running / cfg.log_every
            avg_js = running_js / cfg.log_every
            acc = running_hits / running_n if running_n else None
            t = tctl()
            print(f"step {step:5d}  loss {avg:.4f}" + (f"  js {avg_js:.4f}" if js_value is not None else "")
                  + (f"  acc {acc:.3f}" if acc is not None else "")
                  + f"  {time.time() - t0:.0f}s" + (f"  tctl {t:.0f}°C" if t is not None else ""), flush=True)
            if writer:
                writer.add_scalar("train/loss", avg, step)
                if acc is not None:
                    writer.add_scalar("train/accuracy", acc, step)
                if js_value is not None:
                    writer.add_scalar("train/js", avg_js, step)
                for i, g in enumerate(opt.param_groups):
                    writer.add_scalar(f"train/lr_group{i}", g["lr"], step)
                if t is not None:
                    writer.add_scalar("sys/tctl_c", t, step)
            running, running_js, running_hits, running_n = 0.0, 0.0, 0, 0
        if step % cfg.eval_every == 0 or step == cfg.steps:
            do_eval(step, ce_value, final=step == cfg.steps)
        if on_checkpoint and cfg.save_every > 0 and step % cfg.save_every == 0 and step < cfg.steps:
            on_checkpoint(step)
        if on_state and (step == end or (cfg.save_every > 0 and step % cfg.save_every == 0)):
            on_state(capture_state(step))
    if log_f:
        log_f.close()
    return history
