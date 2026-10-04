"""可训练的部分: attention 上的 LoRA (或主干全参, 见 TRAINABLE) + 嵌入矩阵里放开的行 (256 个 D 行 + 类型 token 行).
其余冻结.

改的是路由 (哪一行匹配) 和槽标记 (D-token 的向量), 知识那部分不碰.

放开的行拆成一个独立的小参数 rows (259 × hidden), 整张嵌入矩阵冻死.
SlotEmbedding 在查表结果上把这几行盖上去, SlotHead 在 logits 上把这几列换掉,
两端共用同一个 rows —— 与 tie_word_embeddings 的语义一致 (菜单里读进来的向量
和答案位置上打分的向量是同一个), 但 AdamW 只给 259 行存状态, 而非整张 151936 行
(fp32 两份矩, 1.2 GiB, 其中 99.8% 永远是零). 之前用整张矩阵 requires_grad + 梯度 hook
清零的做法, 训练效果相同, 显存差在这里.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from sors.core.batch import length_groups, trim_left_padding

ATTN_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_TARGETS = ["gate_proj", "up_proj", "down_proj"]

# 放开的范围. 前三档挂 LoRA (d-only 什么都不挂), 第四档 full 是主干全参.
LORA_TARGETS: dict[str, list[str]] = {
    "d-only": [],  # 基模全冻, 只训 rows —— 纯读出
    "attn": ATTN_TARGETS,
    "attn-mlp": ATTN_TARGETS + MLP_TARGETS,
}
# full: 每一层的全部权重 (attention、MLP、norm) 加最后的 norm 都放开, 不套 LoRA; 词嵌入与输出层的那张
# 151936 行的矩阵照旧冻结、只放 D 行, 与 LoRA 各档只差「主干怎么改」这一个变量.
# 0.6B 的主干约 4.4 亿参数, AdamW 状态加 fp32 主权重 (training.loop.Fp32Master) 约 7 GB, 8GB 卡放不下, 在 A100 上跑.
# 早先不做全参还有一个理由: 它让模型有能力记住数据集的事实, 留出类的成绩就不再说明泛化.
# 现在的评估集 (Banking77 / MASSIVE / BoolQ) 整个不进训练, 这一条不再拦着.
TRAINABLE = (*LORA_TARGETS, "full", "decision-only")


class SlotEmbedding(nn.Module):
    """冻结的整张嵌入 + 一个可训的 rows (n × hidden), 查表后把 ids 对应的位置换成 rows."""

    def __init__(self, base: nn.Embedding, ids: list[int]):
        super().__init__()
        self.base = base
        base.weight.requires_grad_(False)
        self.register_buffer("ids", torch.tensor(ids, device=base.weight.device), persistent=False)
        # 空行的初值是随机的 (基模从没训过它们); 从已有嵌入的均值起步, 让第一步就在合理的尺度上
        with torch.no_grad():
            mu = base.weight[: min(ids)].mean(0)
            init = mu + 0.01 * torch.randn(len(ids), base.weight.shape[1], device=mu.device, dtype=mu.dtype)
        self.rows = nn.Parameter(init)
        # 位置 -> rows 里的行号; 不在 ids 里的是 -1
        lut = torch.full((base.weight.shape[0],), -1, dtype=torch.long, device=base.weight.device)
        lut[self.ids] = torch.arange(len(ids), device=lut.device)
        self.register_buffer("lut", lut, persistent=False)

    @property
    def weight(self) -> torch.Tensor:  # 让读 embed.weight 的外部代码仍拿到整张矩阵
        return self.base.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.base(input_ids)
        pos = self.lut[input_ids]
        indices = (pos.flatten() >= 0).nonzero().flatten()
        if indices.numel():
            out = out.clone(memory_format=torch.contiguous_format)
            # Keep advanced-index backward's low-precision accumulation for
            # repeated token rows; index_select uses a different reduction.
            replacements = self.rows[pos.flatten().index_select(0, indices)].to(out.dtype)
            out.view(-1, out.shape[-1]).index_copy_(0, indices, replacements)
        return out


class SlotHead(nn.Module):
    """冻结的 h @ Wᵀ, 再把 ids 那几列换成 h @ rowsᵀ. rows 与 SlotEmbedding 的是同一个张量."""

    def __init__(self, base: nn.Module, emb: SlotEmbedding):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        self.emb = emb  # 不复制 rows, 引用同一个 Parameter

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.base(h).clone()
        logits[..., self.emb.ids] = (h @ self.emb.rows.T.to(h.dtype)).to(logits.dtype)
        return logits


def prepare_model(
    lm, train_ids: list[int], lora_r: int, lora_alpha: int, lora_dropout: float,
    trainable: str = "attn", grad_ckpt: bool = False,
    decision=None,
):
    """train_ids: 嵌入矩阵里放开的行 —— 256 个 D 行, 加上类型 token 行.
    trainable: TRAINABLE 之一. full 时 lora_r / lora_alpha / lora_dropout 不起作用.

    grad_ckpt: 反传时逐层重算前向, 不存激活. 实测 batch 8 × 512 token 的激活从 5+ GiB 降到 0.58 GiB,
    代价约 +30% 时间. 8GB 卡上 BoolQ passage 进 batch 8 必须开.
    """
    if trainable not in TRAINABLE:
        raise ValueError(f"unknown trainable {trainable!r}; expected one of {TRAINABLE}")
    if trainable == "decision-only" and decision is None:
        raise ValueError("decision-only requires a decision architecture")
    old_vocab = lm.get_input_embeddings().weight.shape[0]
    required_vocab = max(train_ids) + 1
    if required_vocab > old_vocab:
        # Some tokenizers leave fewer spare rows than the slot tokens need. Keep
        # existing vocabularies intact, including their unused output columns.
        lm.resize_token_embeddings(required_vocab, mean_resizing=False)
        # Frozen rows are reconstructed from the base model when loading a
        # checkpoint. Deterministic tails also cover gaps between train_ids.
        with torch.no_grad():
            for layer in (lm.get_input_embeddings(), lm.get_output_embeddings()):
                layer.weight[old_vocab:].copy_(layer.weight[:old_vocab].mean(0))
    for p in lm.parameters():
        p.requires_grad_(False)
    emb = SlotEmbedding(lm.get_input_embeddings(), train_ids)
    lm.set_input_embeddings(emb)
    lm.lm_head = SlotHead(lm.lm_head, emb)
    if grad_ckpt and decision is None:
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if trainable == "full":
        vocab = {id(emb.base.weight), *(id(p) for p in lm.lm_head.base.parameters())}
        for p in lm.parameters():
            if id(p) not in vocab:
                p.requires_grad_(True)
        m = lm
    elif trainable == "decision-only" or not LORA_TARGETS[trainable]:
        m = lm
    else:
        cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=LORA_TARGETS[trainable], bias="none", task_type="CAUSAL_LM",
        )
        m = get_peft_model(lm, cfg)
        emb.rows.requires_grad_(True)  # get_peft_model 会把 LoRA 之外的全部冻上, rows 要在它之后放开
    if grad_ckpt and decision is None:
        # checkpoint 段的输入必须 requires_grad, 否则反传在段边界断掉、LoRA 收不到梯度
        m.enable_input_require_grads()
    if decision is not None:
        from sors.core.decision import DecisionConfig, DecisionModel
        cfg = DecisionConfig(**decision) if isinstance(decision, dict) else decision
        adapter = ({"trainable": "decision-only", "lora_r": None, "lora_alpha": None}
                   if trainable == "decision-only" else adapter_config(m))
        return DecisionModel(m, cfg, adapter, grad_ckpt)
    return m


def adapter_config(m) -> dict:
    """m 放开的范围, 键名与 prepare_model 的参数同名: 哪一档 (TRAINABLE 之一)、LoRA 的 rank、alpha.
    没套 peft 的看 rows 之外还有没有可训参数: 有就是 full, 没有就是 d-only; 两者 rank 与 alpha 无意义记 None."""
    if getattr(m, "decision_config", None) is not None:
        return dict(m.adapter)
    peft_cfg = getattr(m, "peft_config", {}).get("default")
    if peft_cfg is None:
        full = any(p.requires_grad for n, p in m.named_parameters() if not n.endswith(".rows"))
        return {"trainable": "full" if full else "d-only", "lora_r": None, "lora_alpha": None}
    targets = set(peft_cfg.target_modules)
    name = next((k for k, v in LORA_TARGETS.items() if v and set(v) == targets), None)
    if name is None:
        raise ValueError(f"LoRA targets {sorted(targets)} match none of {sorted(LORA_TARGETS)}")
    return {"trainable": name, "lora_r": peft_cfg.r, "lora_alpha": peft_cfg.lora_alpha}


def trainable_param_groups(m, lr_lora: float, lr_embed: float) -> list[dict]:
    """两组: 主干 (LoRA 权重; full 时是主干全部权重) 用 lr_lora, rows 用 lr_embed."""
    body, embed = [], []
    for n, p in m.named_parameters():
        if not p.requires_grad:
            continue
        (embed if n.endswith(".rows") else body).append(p)
    return [{"params": body, "lr": lr_lora}, {"params": embed, "lr": lr_embed}]


def last_logits(m, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """只算最后一个位置的 logits, (B, V). logits_to_keep=1 绕开 (B, L, V) 的大张量."""
    out = m(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=1)
    return out.logits[:, -1, :].float()


def grouped_last_logits(m, input_ids: torch.Tensor, attention_mask: torch.Tensor, n) -> torch.Tensor:
    """与 last_logits 相同的 (B, V), 但把批按真实长度分成 n 组 (batch.length_groups) 各自前向,
    每组只补齐到组里最长的那条, 再按原来的行序拼回. 左填充下答案位置总是最后一列, RoPE 只看相对位置,
    所以每一行的结果与整批一次前向相同 (浮点误差内). 各组的计算图都留着, 调用方照旧对整批的 loss 反传一次.
    n == 1 就是 last_logits 本身. n 也可以是调用方排好的 [(行, 宽度), ...] (见 decision_logits)."""
    if not isinstance(n, int):
        parts = [last_logits(m, input_ids[g][:, -w:], attention_mask[g][:, -w:]) for g, w in n]
        return torch.cat(parts)[torch.argsort(torch.cat([g for g, _ in n]))]
    groups = length_groups(attention_mask, n)
    if len(groups) == 1:
        return last_logits(m, input_ids, attention_mask)
    parts = [last_logits(m, *trim_left_padding(input_ids[g], attention_mask[g])) for g in groups]
    order = torch.cat(groups)
    back = torch.empty_like(order)
    back[order] = torch.arange(len(order), device=order.device)
    return torch.cat(parts)[back]


def select_batch(batch: dict, rows, width: int | None = None) -> dict:
    """Select examples and trim text padding while preserving option coordinates.

    width: the longest selected text, when the caller already knows it on the
    CPU; trimming then reads nothing back from the device.
    """
    out = {k: v[rows] for k, v in batch.items()}
    if width is None:
        ids, mask = trim_left_padding(out["input_ids"], out["attention_mask"])
    else:
        ids, mask = out["input_ids"][:, -width:], out["attention_mask"][:, -width:]
    removed = out["input_ids"].shape[1] - ids.shape[1]
    out.update(input_ids=ids, attention_mask=mask)
    if "option_positions" in out:
        pos = out["option_positions"]
        out["option_positions"] = torch.where(pos >= 0, pos - removed, pos)
    return out


def decision_logits(m, batch: dict, groups=1) -> torch.Tensor:
    """Group complete architecture-aware batches; return logits in caller order.

    groups: a count for batch.length_groups, or a planned [(rows, width), ...]
    such as batch.token_groups produces, with each group's longest text known.
    """
    if not isinstance(groups, int):
        parts = [m.forward_batch(select_batch(batch, rows, width)) for rows, width in groups]
        return torch.cat(parts)[torch.argsort(torch.cat([rows for rows, _ in groups]))]
    rows = length_groups(batch["attention_mask"], groups)
    if len(rows) == 1:
        return m.forward_batch(batch)
    parts = [m.forward_batch(select_batch(batch, group)) for group in rows]
    return torch.cat(parts)[torch.argsort(torch.cat(rows))]
