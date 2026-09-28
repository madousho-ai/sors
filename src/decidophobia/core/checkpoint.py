"""存档读写: 训练只存会变的部分 (LoRA 权重或 full 的主干权重 + 放开的嵌入行), 基模照 model_id 重新加载.

写的是 safetensors, 元数据三段 JSON (d_ids / 训练 config / 放开的范围); 改用 safetensors 之前的 trained.pt 照样读.
训练 (scripts/train.py)、评估脚本、推理服务都从这里还原模型.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import torch

from decidophobia.core.model import adapter_config, prepare_model


def save_trained(m, train_ids: list[int], cfg: TrainConfig, path) -> None:
    """只存会变的部分, 写成 safetensors: 张量是 rows 之外的全部可训参数 (LoRA 权重; full 时是主干的全部权重,
    键名照 named_parameters) 与放开的嵌入行 d_embed (D 行 + 类型行); 元数据是三段 JSON —— d_ids、训练 config、
    adapter (放开的范围与 LoRA 的形状, 见 model.adapter_config, 装档前照它搭空壳). 基模照 model_id 重新加载."""
    from safetensors.torch import save_file

    rows = m.get_input_embeddings().rows
    tensors = {n: p.detach().cpu().contiguous() for n, p in m.named_parameters()
               if p.requires_grad and not n.endswith(".rows")}
    tensors["d_embed"] = rows.detach().cpu().contiguous()
    meta = {"d_ids": train_ids, "config": asdict(cfg), "adapter": adapter_config(m)}
    save_file(tensors, str(path), metadata={k: json.dumps(v) for k, v in meta.items()})


LEGACY_ADAPTER = {"trainable": "attn", "lora_r": 8, "lora_alpha": 16}


def _is_safetensors(path) -> bool:
    """safetensors 开头是 8 字节的头长度, 紧跟 JSON 头的 '{'; torch.save 的档是 zip, 以 'PK' 开头."""
    with open(path, "rb") as f:
        return f.read(9)[8:] == b"{"


def read_checkpoint(path) -> dict:
    """读 save_trained 的档, 还成 {"lora", "d_embed", "d_ids", "config", "adapter"}. "lora" 是 d_embed 之外的
    全部张量, full 的档里是主干权重; 键名沿用旧档的叫法.
    按文件头认格式: 改用 safetensors 之前的档是 torch.save 的 trained.pt, 照读; 更早的档没有 "adapter" 这一项."""
    if not _is_safetensors(path):
        return torch.load(path, map_location="cpu")
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        ck = {k: json.loads(v) for k, v in f.metadata().items()}
        ck["lora"] = {k: f.get_tensor(k) for k in f.keys() if k != "d_embed"}
        ck["d_embed"] = f.get_tensor("d_embed")
    return ck


def checkpoint_adapter(path) -> dict:
    """档里记的放开范围与 LoRA 形状: {"trainable", "lora_r", "lora_alpha"}, 可以直接 ** 进 prepare_model.
    没记的是加这一项之前的档, 那些训练全是 LEGACY_ADAPTER."""
    return read_checkpoint(path).get("adapter", LEGACY_ADAPTER)


def prepare_from_checkpoint(lm, train_ids: list[int], path, lora_dropout: float = 0.0):
    """只拿基模和一份存档还原训练好的模型: 照档里记的放开范围 prepare_model, 再 load_trained.
    返回 (模型, 档里的训练 config). dropout 只在训练时生效, 评估用 0."""
    m = prepare_model(lm, train_ids, lora_dropout=lora_dropout, **checkpoint_adapter(path))
    return m, load_trained(m, train_ids, path)


def load_trained(m, train_ids: list[int], path) -> dict:
    """把 save_trained 存的权重 (LoRA 或 full 的主干) 和嵌入行灌回 prepare_model 之后的模型. 返回存档里的 config.
    safetensors 与旧的 trained.pt 都认, 见 read_checkpoint.

    档里的 ids 允许是模型 train_ids 的前缀: 类型 token 加进来之前的档只有 256 个 D 行,
    那 3 行当时不在提示里、梯度为零, 留在初始化就是那次训练的真实状态. 多出的行原样不动.

    模型的放开范围与 LoRA 形状必须与档里记的相同. 只差 alpha 时张量形状全对得上、拷贝不报错,
    缩放却是错的, 所以在这里对一遍.
    """
    ck = read_checkpoint(path)
    want, got = ck.get("adapter", LEGACY_ADAPTER), adapter_config(m)
    if want != got:
        raise ValueError(f"checkpoint was trained with {want}, this model has {got}")
    n = len(ck["d_ids"])
    if ck["d_ids"] != train_ids[:n]:
        raise ValueError(f"checkpoint's {n} trainable embedding rows are not a prefix of this model's {len(train_ids)}")
    params = dict(m.named_parameters())
    missing = [n for n in ck["lora"] if n not in params]
    if missing:
        raise ValueError(f"{len(missing)} tensors in checkpoint have no home in this model, e.g. {missing[0]}")
    with torch.no_grad():
        for name, t in ck["lora"].items():
            params[name].copy_(t.to(params[name].dtype))
        rows = m.get_input_embeddings().rows
        rows[:n].copy_(ck["d_embed"].to(rows.dtype).to(rows.device))
    return ck["config"]
