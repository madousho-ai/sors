"""candidate 的打包前向: 一个前缀 (材料 + 问题) 只算一次, 它的全部选项分支在同一次前向里接着跑, 可以反传.

语义与逐选项完整编码相同 (每个分支 = 前缀 + 选项 + 读出 token, 单独过一遍主干), 但前缀不复制 K 份:

- 全部 token 排成一条流: 例子 1 的前缀, 它的分支 1..K, 例子 2 的前缀, ...
- 逐 token 的部分 (各个 norm, MLP, 投影) 直接在流上算.
- 全注意力层: 按例子摆成行, 掩码让前缀因果自看, 分支只看本例前缀和自己 (因果);
  RoPE 位置在分支里从前缀长度起算, 每个分支看到的位置与单独编码时相同.
- DeltaNet 层: 先对各前缀跑变长 kernel 拿到末尾的递归状态, 再当初始状态复制给本例每个分支, 分支再跑一次变长 kernel.
  短卷积在每段前接上下文: 前缀前接零 (等于因果零填充), 分支前接本例前缀最后 kernel-1 列的卷积输入.
  梯度经初始状态与卷积上下文回到前缀.

只支持 Qwen3.5 文本主干; DeltaNet 用 fla 的 kernel, 要 CUDA.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def supports_packed(decoder) -> bool:
    cfg = decoder.config
    return str(getattr(cfg, "model_type", "")).startswith("qwen3_5") and \
        set(getattr(cfg, "layer_types", ())) <= {"full_attention", "linear_attention"}


class _Plan:
    """流的布局. 段的顺序: 每个例子先前缀, 再各个分支 (按 valid.nonzero() 的顺序)."""

    def __init__(self, batch, valid, device):
        rows = [tuple(x) for x in valid.nonzero().tolist()]
        by_row: dict[int, list[int]] = {}
        for r, c in rows:
            by_row.setdefault(r, []).append(c)
        tokens, pos = [], []
        self.seg_len, self.seg_kind, self.branch_prefix = [], [], []
        self.example_len = []
        readout = {}
        start = 0
        for r in sorted(by_row):
            prefix = batch["input_ids"][r][batch["attention_mask"][r].bool()][:int(batch["prefix_lengths"][r])]
            p = len(prefix)
            ex_start, prefix_seg = start, -1
            if p:
                tokens.append(prefix)
                pos.append(torch.arange(p, device=device))
                prefix_seg = len(self.seg_len)
                self.seg_len.append(p), self.seg_kind.append(0), self.branch_prefix.append(-1)
                start += p
            for c in by_row[r]:
                n = int(batch["suffix_attention_mask"][r, c].sum())
                if n < 1:
                    raise ValueError("candidate continuation needs a readout token")
                tokens.append(batch["suffix_input_ids"][r, c, -n:])
                pos.append(torch.arange(p, p + n, device=device))
                self.seg_len.append(n), self.seg_kind.append(1), self.branch_prefix.append(prefix_seg)
                start += n
                readout[(r, c)] = start - 1
            self.example_len.append(start - ex_start)
        self.ids = torch.cat(tokens)
        self.pos = torch.cat(pos)
        self.readout = torch.tensor([readout[k] for k in rows], device=device)
        lens = torch.tensor(self.seg_len, device=device)
        self.seg_start = torch.cat([lens.new_zeros(1), lens.cumsum(0)[:-1]]).tolist()
        self.seg_id = torch.repeat_interleave(torch.arange(len(self.seg_len), device=device), lens)
        self.prefix_segs = [i for i, k in enumerate(self.seg_kind) if k == 0]
        self.branch_segs = [i for i, k in enumerate(self.seg_kind) if k == 1]
        self._rows = None

    def rows(self, device):
        """全注意力用的行布局: 每个例子一行, 右填充; 返回 (流下标, 有效位, 注意力掩码)."""
        if self._rows is None:
            width = max(self.example_len)
            lens = torch.tensor(self.example_len, device=device)
            starts = torch.cat([lens.new_zeros(1), lens.cumsum(0)[:-1]])
            col = torch.arange(width, device=device)
            index = (starts[:, None] + col[None]).clamp(max=len(self.ids) - 1)
            live = col[None] < lens[:, None]
            seg = self.seg_id[index].masked_fill(~live, -1)
            kind = torch.tensor(self.seg_kind, device=device)[seg.clamp_min(0)]
            same = seg[:, :, None] == seg[:, None, :]
            from_prefix = (kind[:, None, :] == 0) & live[:, None, :]
            causal = col[:, None] >= col[None, :]
            mask = causal[None] & (same | from_prefix) & live[:, None, :] & live[:, :, None]
            mask |= torch.eye(width, dtype=torch.bool, device=device)[None]  # 填充位自看, 免得整行被屏蔽
            self._rows = (index, live, mask[:, None])
        return self._rows


def _full_attention(attn, h, cos, sin, plan):
    """Qwen3_5Attention 的数学, 掩码由 plan 给出. h: (T, hidden)."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    index, live, mask = plan.rows(h.device)
    x = h[index]  # (B, L, hidden)
    lead = x.shape[:-1]
    q, gate = torch.chunk(attn.q_proj(x).view(*lead, -1, attn.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(*lead, -1)
    q = attn.q_norm(q.reshape(*lead, -1, attn.head_dim)).transpose(1, 2)
    k = attn.k_norm(attn.k_proj(x).view(*lead, -1, attn.head_dim)).transpose(1, 2)
    v = attn.v_proj(x).view(*lead, -1, attn.head_dim).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos[index], sin[index])
    rep = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=attn.scaling)
    out = out.transpose(1, 2).reshape(*lead, -1) * torch.sigmoid(gate)
    return attn.o_proj(out)[live]  # 行主序的有效位正好是流的顺序


def _conv_with_context(layer, qkv, plan):
    """每段的因果短卷积 + SiLU, 段前接上下文 (前缀接零, 分支接本例前缀末尾). qkv: (T, C)."""
    width = layer.conv_kernel_size - 1
    weight = layer.conv1d.weight  # (C, 1, k)
    zeros = qkv.new_zeros(width, qkv.shape[1])
    pieces = []
    for s, (start, n) in enumerate(zip(plan.seg_start, plan.seg_len)):
        p = plan.branch_prefix[s]
        if p < 0:
            ctx = zeros
        else:
            end = plan.seg_start[p] + plan.seg_len[p]
            ctx = qkv[max(plan.seg_start[p], end - width):end]
            if len(ctx) < width:
                ctx = torch.cat([zeros[:width - len(ctx)], ctx])
        pieces += [ctx, qkv[start:start + n]]
    stream = torch.cat(pieces).T[None].to(weight.dtype)  # (1, C, T + width * 段数)
    out = F.conv1d(stream, weight, layer.conv1d.bias, groups=stream.shape[1])[0].T
    chunks, at = [], 0  # 输出第 j 行对应 stream 第 j + width 列
    for n in plan.seg_len:
        chunks.append(out[at:at + n])
        at += width + n
    return F.silu(torch.cat(chunks)).to(qkv.dtype)


def _deltanet(layer, h, plan):
    """Qwen3_5GatedDeltaNet 的数学: 前缀跑一次, 分支从前缀末尾的递归状态出发."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    T = h.shape[0]
    qkv = _conv_with_context(layer, layer.in_proj_qkv(h), plan)
    z = layer.in_proj_z(h).reshape(T, -1, layer.head_v_dim)
    beta = layer.in_proj_b(h).sigmoid()
    g = -layer.A_log.float().exp() * F.softplus(layer.in_proj_a(h).float() + layer.dt_bias)
    q, k, v = torch.split(qkv, [layer.key_dim, layer.key_dim, layer.value_dim], dim=-1)
    q = q.reshape(T, -1, layer.head_k_dim)
    k = k.reshape(T, -1, layer.head_k_dim)
    v = v.reshape(T, -1, layer.head_v_dim)
    if layer.num_v_heads // layer.num_k_heads > 1:
        rep = layer.num_v_heads // layer.num_k_heads
        q, k = q.repeat_interleave(rep, dim=1), k.repeat_interleave(rep, dim=1)

    def run(segs, init):
        idx = torch.cat([torch.arange(plan.seg_start[s], plan.seg_start[s] + plan.seg_len[s], device=h.device)
                         for s in segs])
        cu = torch.tensor([0] + [plan.seg_len[s] for s in segs], device=h.device).cumsum(0)
        o, final = chunk_gated_delta_rule(q[idx][None], k[idx][None], v[idx][None], g=g[idx][None],
                                          beta=beta[idx][None], initial_state=init, output_final_state=True,
                                          use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
        return idx, o[0], final

    out = v.new_zeros(v.shape)
    finals = {}
    if plan.prefix_segs:
        idx, o, final = run(plan.prefix_segs, None)
        out = out.index_put((idx,), o.to(out.dtype))
        finals = {s: final[i] for i, s in enumerate(plan.prefix_segs)}
    if plan.branch_segs:
        shape = (v.shape[1], q.shape[-1], v.shape[-1])
        zero = v.new_zeros(shape, dtype=torch.float32)
        init = torch.stack([finals[p] if p >= 0 else zero for p in (plan.branch_prefix[s] for s in plan.branch_segs)])
        idx, o, _ = run(plan.branch_segs, init)
        out = out.index_put((idx,), o.to(out.dtype))
    core = layer.norm(out.reshape(-1, layer.head_v_dim), z.reshape(-1, layer.head_v_dim))
    return layer.out_proj(core.reshape(T, -1))


def _layer(layer, h, cos, sin, plan):
    residual = h
    x = layer.input_layernorm(h)
    if layer.block_type == "linear_attention":
        x = _deltanet(layer.linear_attn, x, plan)
    else:
        x = _full_attention(layer.self_attn, x, cos, sin, plan)
    h = residual + x
    return h + layer.mlp(layer.post_attention_layernorm(h))


def packed_candidate_hidden(decoder, batch, valid, checkpointed: bool = False):
    """每个有效选项读出 token 上的最后一层隐藏状态 (final norm 之后), 顺序同 valid.nonzero()."""
    if not supports_packed(decoder):
        raise ValueError("packed candidate encoding supports Qwen3.5 text backbones only")
    plan = _Plan(batch, valid, batch["input_ids"].device)
    h = decoder.embed_tokens(plan.ids)  # (T, hidden)
    cos, sin = decoder.rotary_emb(h[None], plan.pos[None, None].expand(3, 1, -1))
    cos, sin = cos[0], sin[0]  # (T, rotary dim)
    for layer in decoder.layers[:decoder.config.num_hidden_layers]:
        if checkpointed and torch.is_grad_enabled():
            h = checkpoint(_layer, layer, h, cos, sin, plan, use_reentrant=False)
        else:
            h = _layer(layer, h, cos, sin, plan)
    return decoder.norm(h)[plan.readout]
