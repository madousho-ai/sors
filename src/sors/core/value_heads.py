"""Qwen3.5 线性注意力 (Gated DeltaNet) 层的 V 头扩展.

每个 V 头有一个 head_k_dim x head_v_dim 的递归状态; V 头多于 k 头时, V 头 j 共用 k 头 j // (V/K) 的 q、k
(transformers 里是 repeat_interleave). 扩到 n 个头时, 新的头 j 复制旧头 j // f (f = n / 旧头数) 的全部参数:
v 的投影行和短卷积通道、门 z 的投影行、写入强度 b 与衰减 a 的投影行、A_log、dt_bias. 旧头 j // f 落在
新的 j = f * (j // f), 它在 out_proj 里的列原样搬过去; 其余新头的列置零.

这样刚扩完的模型逐 token 与原模型相同 (新头的输出乘的是零列); 新头本身的输出非零, out_proj 的新列
一开始就收得到梯度. v 与 out_proj 同时置零的话两边梯度都恒为零, 新头永远学不动.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# 扩过的模型在文本 config 上记扩到几个头; 导出时随 config 一起存下, 存档的 adapter 也照它记.
EXPANDED = "sors_linear_value_heads"


def _linear(old: nn.Linear, rows: torch.Tensor) -> nn.Linear:
    new = nn.Linear(old.in_features, rows.shape[0], bias=False, device=rows.device, dtype=rows.dtype)
    new.weight.data.copy_(rows)
    return new


def _expand_layer(layer, heads: int) -> None:
    old, dk, dv = layer.num_v_heads, layer.head_k_dim, layer.head_v_dim
    f = heads // old
    source = torch.arange(heads, device=layer.A_log.device) // f  # 新头 j 复制旧头 j // f
    kept = torch.arange(heads, device=source.device) % f == 0  # 原来的头, out_proj 的列原样保留

    def per_head(t: torch.Tensor, width: int) -> torch.Tensor:  # 按头复制前 old*width 行
        return t.view(old, width, *t.shape[1:])[source].reshape(heads * width, *t.shape[1:])

    with torch.no_grad():
        qk, v = layer.in_proj_qkv.weight.split([2 * layer.key_dim, layer.value_dim])
        layer.in_proj_qkv = _linear(layer.in_proj_qkv, torch.cat([qk, per_head(v, dv)]))
        conv = layer.conv1d.weight
        conv_qk, conv_v = conv.split([2 * layer.key_dim, layer.value_dim])
        weight = torch.cat([conv_qk, per_head(conv_v, dv)])
        layer.conv1d = nn.Conv1d(weight.shape[0], weight.shape[0], kernel_size=layer.conv_kernel_size,
                                 groups=weight.shape[0], padding=layer.conv_kernel_size - 1, bias=False,
                                 device=weight.device, dtype=weight.dtype)
        layer.conv1d.weight.copy_(weight)
        layer.in_proj_z = _linear(layer.in_proj_z, per_head(layer.in_proj_z.weight, dv))
        layer.in_proj_b = _linear(layer.in_proj_b, layer.in_proj_b.weight[source])
        layer.in_proj_a = _linear(layer.in_proj_a, layer.in_proj_a.weight[source])
        layer.A_log = nn.Parameter(layer.A_log.data[source].clone())
        layer.dt_bias = nn.Parameter(layer.dt_bias.data[source].clone())
        out = layer.out_proj.weight  # (hidden, old * dv)
        cols = out.view(out.shape[0], old, dv)[:, source] * kept.view(1, heads, 1).to(out.dtype)
        new_out = nn.Linear(heads * dv, out.shape[0], bias=False, device=out.device, dtype=out.dtype)
        new_out.weight.copy_(cols.reshape(out.shape[0], heads * dv))
        layer.out_proj = new_out
    layer.num_v_heads = heads
    layer.value_dim = heads * dv
    layer.conv_dim = 2 * layer.key_dim + layer.value_dim


def expand_linear_value_heads(lm, heads: int) -> None:
    """把 lm 每个线性注意力层的 V 头扩到 heads 个, 原地改. 已经是 heads 个时什么都不动.
    heads 必须是现有头数的整数倍 (于是也是 k 头数的整数倍)."""
    config = lm.config.get_text_config()
    old = config.linear_num_value_heads
    if heads == old:
        return
    if heads < old or heads % old:
        raise ValueError(f"linear value heads can only grow by a whole multiple of {old}; got {heads}")
    for decoder_layer in lm.model.layers:
        if getattr(decoder_layer, "block_type", None) == "linear_attention":
            _expand_layer(decoder_layer.linear_attn, heads)
    config.linear_num_value_heads = heads
    setattr(config, EXPANDED, heads)
