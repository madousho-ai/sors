"""candidate 的隐式推理步 (Coconut 式): 前缀 (材料 + 问题) 之后接 n 个隐式位置, 各选项分支挂在最后一个后面.

第 j 个隐式位置的输入向量是「前缀 + 前 j-1 个隐式位置」最后一个位置的最后一层隐藏状态 (final norm 之后),
不经过词表、不采样. 隐式位置每题只算一遍, 本题全部选项分支共用; 分支之间仍互相看不见.

隐式位置之间是串行依赖, 所以分阶段跑, 每阶段把全部层走完:
  阶段 0  各题前缀
  阶段 j  每题 1 个隐式位置 (j = 1..n)
  末阶段  全部选项分支
每阶段读之前各阶段留下的上下文, 并把自己的追加进去:
  - 全注意力层: 旋转之后的 K / V, 每题一行 (E, 列, 头, 维): 前缀补齐到同一宽度, 隐式位置依次追加成新列.
    旋转位置已经算进 K 里, 列的排法不影响注意力; 有效列由掩码给出. 只拼接、不按下标取, 反传没有原子加.
  - DeltaNet 层: 每题末尾的递归状态, 与短卷积要接的最后 conv_width 列卷积输入.
前缀不重算. 梯度经缓存回到前缀与各隐式位置.

段边界在 CPU 上算好, 连同 CPU 副本交给 fla (cu_seqlens_cpu), 每层不必把边界读回主机.

只支持 Qwen3.5 文本主干 (不收带图的批); DeltaNet 用 fla 的 kernel, 要 CUDA.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from sors.core import packed as _packed


class _Stage:
    """一个阶段的布局 (CPU 上算好): 本阶段的段, 每段属于哪一题 (按 0..E-1 编号, 段按题排好), 旋转位置,
    以及全注意力用的行布局 (每题一行: key = 这题的缓存列 + 本阶段这题的全部段; query = 本阶段这题的 token)."""

    def __init__(self, segs, owner, offsets, cache_valid, conv_width, device):
        if any(a > b for a, b in zip(owner, owner[1:])):
            raise ValueError("stage segments must be grouped by example in order")
        put = lambda t: t.to(device, non_blocking=True)
        lens = [len(s) for s in segs]
        E, A = cache_valid.shape
        self.E, self.A, self.T = E, A, sum(lens)
        self.single = all(n == 1 for n in lens) and owner == list(range(E))  # 隐式阶段: 每题恰好一个 token
        self.ids = put(torch.cat(segs)) if segs and isinstance(segs[0], torch.Tensor) else None
        self.pos = put(torch.cat([torch.arange(o, o + n) for o, n in zip(offsets, lens)]).expand(3, -1))
        starts = torch.tensor([0] + lens).cumsum(0)
        self.seg_start = starts[:-1].tolist()
        self.cu_cpu = starts
        self.cu = put(starts)
        self.bounds = starts.tolist()
        self.owner = put(torch.tensor(owner))
        self.last = put(starts[1:] - 1)  # 每段末尾那个 token: 隐式阶段取它作下一步输入, 末阶段取它作读出
        self.conv_width = conv_width
        tail = [None] * E  # 每题最后一段: 本阶段结束后用它更新这题的卷积尾与递归状态
        for s, e in enumerate(owner):
            tail[e] = s
        self.example_tail = tail
        if not self.single:
            # 短卷积: 每段前接本题缓存里的 conv_width 列 (表 = 本阶段流 + E x w 列缓存)
            conv_index, conv_keep, at = [], [], 0
            for s, (st, n) in enumerate(zip(self.seg_start, lens)):
                e = owner[s]
                conv_index += [self.T + e * conv_width + i for i in range(conv_width)] + list(range(st, st + n))
                conv_keep += range(at, at + n)  # conv 输出第 j 行 = 输入第 j..j+w 列, 即段内第 j - at 个 token
                at += conv_width + n
            self.conv_gather = _packed._Index(torch.tensor(conv_index), put)
            self.conv_keep = put(torch.tensor(conv_keep))
            if any(t is None for t in tail):
                raise ValueError("every example needs at least one segment in a cache-writing stage")
            self.tail_seg = put(torch.tensor([t if t is not None else 0 for t in tail]))
            self.tail_end = put(starts[1:][torch.tensor([t if t is not None else 0 for t in tail])])
        by_e = [[] for _ in range(E)]
        for s, e in enumerate(owner):
            by_e[e].append(s)
        Lq = max(1, max(sum(lens[s] for s in segs_e) for segs_e in by_e))
        q_index = torch.zeros(E, Lq, dtype=torch.long)
        q_live = torch.zeros(E, Lq, dtype=torch.bool)
        seg_of = torch.full((E, Lq), -1, dtype=torch.long)
        for e in range(E):
            at = 0
            for s in by_e[e]:
                n = lens[s]
                q_index[e, at:at + n] = torch.arange(self.seg_start[s], self.seg_start[s] + n)
                q_live[e, at:at + n] = True
                seg_of[e, at:at + n] = s
                at += n
        causal = torch.arange(Lq)[:, None] >= torch.arange(Lq)[None]
        self_mask = causal[None] & (seg_of[:, :, None] == seg_of[:, None, :]) & (seg_of[:, :, None] >= 0)
        self_mask |= torch.eye(Lq, dtype=torch.bool)[None] & ~q_live[:, :, None]  # 填充位自看, 免得整行被屏蔽
        cache_mask = cache_valid[:, None, :].expand(E, Lq, A) & q_live[:, :, None]
        self.mask = put(torch.cat([cache_mask, self_mask], 2)[:, None])
        self.q_index, self.q_live = put(q_index), put(q_live)
        self.own_valid = q_live  # 本阶段写进缓存的列 (CPU), 只对写缓存的阶段有意义


class _Cache:
    """每层的上下文: 全注意力层每题一行的 K / V (E, 列, 头, 维) 与有效列; DeltaNet 层每题的递归状态与卷积尾."""

    def __init__(self, E):
        self.k, self.v = {}, {}
        self.valid = torch.zeros(E, 0, dtype=torch.bool)  # CPU
        self.state, self.tail = {}, {}  # 层号 -> (E, H, K, V) fp32 / (E, w, C)


def _attention(attn, x, cos, sin, stage, past_k, past_v):
    """x: 本阶段的 (T, hidden). 返回 (输出 (T, hidden), 本阶段每题一行的 k, v (E, Lq, kv_heads, head_dim))."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    T = x.shape[0]
    q, gate = torch.chunk(attn.q_proj(x).view(T, -1, attn.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(T, -1)
    q = attn.q_norm(q.reshape(T, -1, attn.head_dim))
    k = attn.k_norm(attn.k_proj(x).view(T, -1, attn.head_dim))
    v = attn.v_proj(x).view(T, -1, attn.head_dim)
    q, k = apply_rotary_pos_emb(q.transpose(0, 1)[None], k.transpose(0, 1)[None], cos[None], sin[None])
    q, k = q[0].transpose(0, 1), k[0].transpose(0, 1)  # (T, heads, d)
    rows_q = q[stage.q_index].transpose(1, 2)  # (E, heads, Lq, d)
    own_k, own_v = k[stage.q_index], v[stage.q_index]  # (E, Lq, kvh, d)
    keys = torch.cat([past_k, own_k], 1) if past_k is not None else own_k
    vals = torch.cat([past_v, own_v], 1) if past_v is not None else own_v
    keys, vals = keys.transpose(1, 2), vals.transpose(1, 2)
    rep = rows_q.shape[1] // keys.shape[1]
    keys, vals = keys.repeat_interleave(rep, 1), vals.repeat_interleave(rep, 1)
    out = F.scaled_dot_product_attention(rows_q, keys, vals, attn_mask=stage.mask, scale=attn.scaling)
    out = out.transpose(1, 2)[stage.q_live].reshape(T, -1) * torch.sigmoid(gate)
    return attn.o_proj(out), own_k, own_v


def _conv(layer, raw, stage, ctx):
    """因果短卷积 + SiLU, 每段前接本题的卷积尾 ctx (E, w, C). 返回 (qkv, 每题新卷积尾 (E, w, C))."""
    weight = layer.conv1d.weight  # (C, 1, w + 1)
    w = stage.conv_width
    if stage.single:  # 每题一个 token: 窗口 = 卷积尾 + 这个 token, 直接加权求和
        window = torch.cat([ctx.to(raw.dtype), raw[:, None]], 1)  # (E, w + 1, C)
        out = (window.to(weight.dtype) * weight[:, 0].T[None]).sum(1)
        if layer.conv1d.bias is not None:
            out = out + layer.conv1d.bias
        return F.silu(out).to(raw.dtype), window[:, 1:]
    E = stage.E
    padded = torch.cat([raw, ctx.reshape(E * w, -1).to(raw.dtype)])
    stream = stage.conv_gather(padded).T[None].to(weight.dtype)
    out = F.conv1d(stream, weight, layer.conv1d.bias, groups=stream.shape[1])[0].T
    qkv = F.silu(out.index_select(0, stage.conv_keep)).to(raw.dtype)
    # 新卷积尾: 每题最后一段的最后 w 列 (段不足 w 时前面接旧尾)
    ext = torch.cat([ctx.to(raw.dtype), raw.new_zeros(E, 0, raw.shape[1])], 1)
    cols = stage.tail_end[:, None] - w + torch.arange(w, device=raw.device)[None]  # 流里的列号, 可能 < 段首
    seg_start = stage.cu.index_select(0, stage.tail_seg)[:, None]
    from_ctx = cols < seg_start
    pick = raw.index_select(0, cols.clamp_min(0).flatten()).view(E, w, -1)
    back = (cols - seg_start + w).clamp(0, w - 1)  # 落在旧尾里时的下标
    old = ext.gather(1, back[..., None].expand(-1, -1, raw.shape[1]))
    return qkv, torch.where(from_ctx[..., None], old, pick)


def _deltanet(layer, x, stage, state, tail):
    """x: (T, hidden). state: (E, H, K, V) fp32 或 None (零); tail: (E, w, C) 或 None (零).
    返回 (输出, 每题末状态 (E, ...), 每题新卷积尾 (E, w, C))."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    T, E, w = x.shape[0], stage.E, stage.conv_width
    raw = layer.in_proj_qkv(x)
    ctx = tail if tail is not None else raw.new_zeros(E, w, raw.shape[1])
    qkv, new_tail = _conv(layer, raw, stage, ctx)
    z = layer.in_proj_z(x).reshape(T, -1, layer.head_v_dim)
    beta = layer.in_proj_b(x).sigmoid()
    g = -layer.A_log.float().exp() * F.softplus(layer.in_proj_a(x).float() + layer.dt_bias)
    q, k, v = torch.split(qkv, [layer.key_dim, layer.key_dim, layer.value_dim], dim=-1)
    q, k, v = q.reshape(T, -1, layer.head_k_dim), k.reshape(T, -1, layer.head_k_dim), v.reshape(T, -1, layer.head_v_dim)
    if layer.num_v_heads // layer.num_k_heads > 1:
        rep = layer.num_v_heads // layer.num_k_heads
        q, k = q.repeat_interleave(rep, dim=1), k.repeat_interleave(rep, dim=1)
    init = state.index_select(0, stage.owner) if state is not None else None
    n = len(stage.bounds) - 1
    per_call = max(1, _packed.KERNEL_GRID_ROWS // v.shape[1])
    outs, finals = [], []
    for s in range(0, n, per_call):
        e = min(n, s + per_call)
        a, b = stage.bounds[s], stage.bounds[e]
        cu, cu_cpu = (stage.cu, stage.cu_cpu) if (s, e) == (0, n) else \
            (stage.cu[s:e + 1] - stage.bounds[s], stage.cu_cpu[s:e + 1] - stage.bounds[s])
        o, final = chunk_gated_delta_rule(q[a:b][None], k[a:b][None], v[a:b][None], g=g[a:b][None], beta=beta[a:b][None],
                                          initial_state=None if init is None else init[s:e],
                                          output_final_state=True, use_qk_l2norm_in_kernel=True,
                                          cu_seqlens=cu, cu_seqlens_cpu=cu_cpu)
        outs.append(o[0])
        finals.append(final)
    out = torch.cat(outs) if len(outs) > 1 else outs[0]
    core = layer.norm(out.reshape(-1, layer.head_v_dim), z.reshape(-1, layer.head_v_dim))
    final = torch.cat(finals) if len(finals) > 1 else finals[0]
    if not stage.single:
        final = final.index_select(0, stage.tail_seg)
    return layer.out_proj(core.reshape(T, -1)), final.float(), new_tail


def _stage_layer(layer, h, cos, sin, stage, past_k, past_v, state, tail):
    residual = h
    x = layer.input_layernorm(h)
    k = v = final = new_tail = None
    if layer.block_type == "linear_attention":
        x, final, new_tail = _deltanet(layer.linear_attn, x, stage, state, tail)
    else:
        x, k, v = _attention(layer.self_attn, x, cos, sin, stage, past_k, past_v)
    h = residual + x
    h = h + layer.mlp(layer.post_attention_layernorm(h))
    return h, k, v, final, new_tail


def _run_stage(decoder, h, stage, cache, checkpointed, write):
    """本阶段的全部层; write 时把本阶段的 K / V 列、末状态、卷积尾记进 cache. 返回 final norm 之后的 h."""
    cos, sin = decoder.rotary_emb(h[None], stage.pos[:, None])
    cos, sin = cos[0], sin[0]
    for i, layer in enumerate(decoder.layers[:decoder.config.num_hidden_layers]):
        args = (layer, h, cos, sin, stage, cache.k.get(i), cache.v.get(i), cache.state.get(i), cache.tail.get(i))
        if checkpointed and torch.is_grad_enabled():
            h, k, v, final, tail = checkpoint(_stage_layer, *args, use_reentrant=False)
        else:
            h, k, v, final, tail = _stage_layer(*args)
        if write:
            if k is not None:
                cache.k[i] = k if i not in cache.k else torch.cat([cache.k[i], k], 1)
                cache.v[i] = v if i not in cache.v else torch.cat([cache.v[i], v], 1)
            if final is not None:
                cache.state[i], cache.tail[i] = final, tail
    if write:
        cache.valid = torch.cat([cache.valid, stage.own_valid], 1)
    return decoder.norm(h)


def latent_candidate_hidden(decoder, batch, valid, steps: int, checkpointed: bool = False):
    """每个有效选项读出 token 上的最后一层隐藏状态 (final norm 之后), 顺序同 valid.nonzero();
    前缀后面先接 steps 个隐式位置."""
    if "image_embeds" in batch or "image_pixels" in batch:
        raise ValueError("latent steps do not support images in the state yet")
    linear = [l.linear_attn for l in decoder.layers if getattr(l, "block_type", "") == "linear_attention"]
    w = linear[0].conv_kernel_size - 1 if linear else 3
    device = batch["input_ids"].device
    keys = ["input_ids", "attention_mask", "prefix_lengths", "suffix_input_ids", "suffix_attention_mask"]
    cpu = {k: batch[k].cpu() for k in keys}
    rows = [tuple(x) for x in valid.cpu().nonzero().tolist()]
    examples = sorted({r for r, _ in rows})
    E = len(examples)
    prefixes = []
    for r in examples:
        p = cpu["input_ids"][r][cpu["attention_mask"][r].bool()][:int(cpu["prefix_lengths"][r])]
        if len(p) == 0:
            raise ValueError("latent steps need a non-empty prefix")
        prefixes.append(p)
    plen = [len(p) for p in prefixes]
    cache = _Cache(E)

    stage = _Stage(prefixes, list(range(E)), [0] * E, cache.valid, w, device)  # 阶段 0: 前缀
    h = _run_stage(decoder, decoder.embed_tokens(stage.ids), stage, cache, checkpointed, True)
    x = h.index_select(0, stage.last)  # (E, hidden): 第 1 个隐式位置的输入
    one = [torch.zeros(1, dtype=torch.long)] * E
    for j in range(1, steps + 1):  # 阶段 j: 每题 1 个隐式位置
        stage = _Stage(one, list(range(E)), [plen[e] + j - 1 for e in range(E)], cache.valid, w, device)
        x = _run_stage(decoder, x, stage, cache, checkpointed, True)
    lengths = cpu["suffix_attention_mask"].sum(-1)  # 末阶段: 各选项分支, 从最后一个隐式位置之后接着数位置
    index = {r: e for e, r in enumerate(examples)}
    segs, owner = [], []
    for r, c in rows:
        n = int(lengths[r, c])
        if n < 1:
            raise ValueError("candidate continuation needs a readout token")
        segs.append(cpu["suffix_input_ids"][r, c, -n:])
        owner.append(index[r])
    stage = _Stage(segs, owner, [plen[e] + steps for e in owner], cache.valid, w, device)
    h = _run_stage(decoder, decoder.embed_tokens(stage.ids), stage, cache, checkpointed, False)
    return h.index_select(0, stage.last)
