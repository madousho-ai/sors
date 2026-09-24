"""可训练的部分: attention 上的 LoRA + 嵌入矩阵里放开的行 (256 个 D 行 + 类型 token 行). 其余冻结.

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

ATTN_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_TARGETS = ["gate_proj", "up_proj", "down_proj"]

# 放开的范围, 三档. 全参不在其中: 0.6B 全参 AdamW 的优化器状态 8GB 卡放不下,
# 更要紧的是它让模型有能力记住数据集的事实, 留出类的成绩就不再说明泛化.
LORA_TARGETS: dict[str, list[str]] = {
    "d-only": [],  # 基模全冻, 只训 rows —— 纯读出
    "attn": ATTN_TARGETS,
    "attn-mlp": ATTN_TARGETS + MLP_TARGETS,
}


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
        hit = pos >= 0
        if hit.any():
            out = out.clone()
            out[hit] = self.rows[pos[hit]].to(out.dtype)
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
):
    """train_ids: 嵌入矩阵里放开的行 —— 256 个 D 行, 加上类型 token 行.

    grad_ckpt: 反传时逐层重算前向, 不存激活. 实测 batch 8 × 512 token 的激活从 5+ GiB 降到 0.58 GiB,
    代价约 +30% 时间. 8GB 卡上 BoolQ passage 进 batch 8 必须开.
    """
    for p in lm.parameters():
        p.requires_grad_(False)
    emb = SlotEmbedding(lm.get_input_embeddings(), train_ids)
    lm.set_input_embeddings(emb)
    lm.lm_head = SlotHead(lm.lm_head, emb)
    if grad_ckpt:
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    targets = LORA_TARGETS[trainable]
    if not targets:
        m = lm
    else:
        cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=targets, bias="none", task_type="CAUSAL_LM",
        )
        m = get_peft_model(lm, cfg)
        emb.rows.requires_grad_(True)  # get_peft_model 会把 LoRA 之外的全部冻上, rows 要在它之后放开
    if grad_ckpt:
        # checkpoint 段的输入必须 requires_grad, 否则反传在段边界断掉、LoRA 收不到梯度
        m.enable_input_require_grads()
    return m


def adapter_config(m) -> dict:
    """m 身上 LoRA 的形状, 键名与 prepare_model 的参数同名: 挂在哪档 (LORA_TARGETS 的键)、rank、alpha.
    没套 peft 的就是 d-only, rank 与 alpha 无意义记 None."""
    peft_cfg = getattr(m, "peft_config", {}).get("default")
    if peft_cfg is None:
        return {"trainable": "d-only", "lora_r": None, "lora_alpha": None}
    targets = set(peft_cfg.target_modules)
    name = next((k for k, v in LORA_TARGETS.items() if v and set(v) == targets), None)
    if name is None:
        raise ValueError(f"LoRA targets {sorted(targets)} match none of {sorted(LORA_TARGETS)}")
    return {"trainable": name, "lora_r": peft_cfg.r, "lora_alpha": peft_cfg.lora_alpha}


def trainable_param_groups(m, lr_lora: float, lr_embed: float) -> list[dict]:
    lora, embed = [], []
    for n, p in m.named_parameters():
        if not p.requires_grad:
            continue
        (embed if n.endswith(".rows") else lora).append(p)
    return [{"params": lora, "lr": lr_lora}, {"params": embed, "lr": lr_embed}]


def last_logits(m, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """只算最后一个位置的 logits, (B, V). logits_to_keep=1 绕开 (B, L, V) 的大张量."""
    out = m(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=1)
    return out.logits[:, -1, :].float()
