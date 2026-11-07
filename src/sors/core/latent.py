"""candidate 的隐式推理步 (Coconut 式): 前缀 (材料 + 问题) 之后接 n 个隐式位置, 各选项分支挂在最后一个后面.

第 j 个隐式位置的输入向量是「前缀 + 前 j-1 个隐式位置」最后一个位置的最后一层隐藏状态 (final norm 之后),
不经过词表、不采样. 隐式位置每题只算一遍, 本题全部选项分支共用; 分支之间仍互相看不见.

隐式位置之间是串行依赖, 所以分阶段跑, 每阶段把全部层走完:
  阶段 0  各题前缀
  阶段 j  每题 1 个隐式位置 (j = 1..n)
  末阶段  全部选项分支
每阶段读之前各阶段留下的上下文, 并把自己的追加进去:
  - 全注意力层: 旋转之后的 K / V, 按流拼接 (每题一条祖先链: 前缀 + 已有隐式位置);
  - DeltaNet 层: 每题末尾的递归状态, 与短卷积要接的最后 conv_width 列卷积输入.
前缀不重算. 梯度经缓存回到前缀与各隐式位置.

只支持 Qwen3.5 文本主干 (不收带图的批); DeltaNet 用 fla 的 kernel, 要 CUDA.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from sors.core import packed as _packed


class _Stage:
    """一个阶段的布局 (CPU 上算好): 本阶段的段, 每段属于哪一题 (按 0..E-1 编号), 旋转位置,
    以及全注意力用的行布局 (每题一行: 先是这题的缓存, 再是本阶段这题的全部段)."""

    def __init__(self, segs, owner, offsets, cache_len, conv_width, device):
        put = lambda t: t.to(device, non_blocking=True)
        if any(a > b for a, b in zip(owner, owner[1:])):
            raise ValueError("stage segments must be grouped by example in order")
        lens = [len(s) for s in segs]
        self.T = sum(lens)
        E = len(cache_len)
        self.E = E
        self.ids = put(torch.cat(segs)) if segs and isinstance(segs[0], torch.Tensor) else None
        self.pos = put(torch.cat([torch.arange(o, o + n) for o, n in zip(offsets, lens)]).expand(3, -1))
        starts = torch.tensor([0] + lens).cumsum(0)
        self.seg_start = starts[:-1].tolist()
        self.cu = put(starts)
        self.owner = put(torch.tensor(owner))
        # 每段末尾那个 token 在本阶段流里的下标 (隐式阶段取它作下一步输入, 末阶段取它作读出)
        self.last = put(starts[1:] - 1)
        # 短卷积: 每段前接本题缓存里的 conv_width 列 (表 = 本阶段流 + E x w 列缓存)
        conv_index, conv_keep, at = [], [], 0
        for s, (st, n) in enumerate(zip(self.seg_start, lens)):
            e = owner[s]
            conv_index += [self.T + e * conv_width + i for i in range(conv_width)] + list(range(st, st + n))
            conv_keep += range(at + conv_width, at + conv_width + n)
            at += conv_width + n
        self.conv_gather = _packed._Index(torch.tensor(conv_index), put)
        self.conv_keep = put(torch.tensor(conv_keep))
        self.conv_width = conv_width
        # 每题最后一段的末尾: 本阶段结束后更新这题的卷积缓存与递归状态 (隐式阶段每题恰好一段)
        self.example_tail = [None] * E
        for s, e in enumerate(owner):
            self.example_tail[e] = s
        # 全注意力的行: 第 e 行的 key = 这题缓存 (cache_len[e]) + 本阶段这题的 token; query = 本阶段这题的 token
        by_e = [[] for _ in range(E)]
        for s, e in enumerate(owner):
            by_e[e].append(s)
        q_rows = [torch.cat([torch.arange(self.seg_start[s], self.seg_start[s] + lens[s]) for s in by_e[e]])
                  if by_e[e] else torch.zeros(0, dtype=torch.long) for e in range(E)]
        q_seg = [torch.cat([torch.full((lens[s],), s) for s in by_e[e]]) if by_e[e] else torch.zeros(0, dtype=torch.long)
                 for e in range(E)]
        Lq = max(1, max(len(r) for r in q_rows))
        A = max(cache_len) if E else 0
        q_index = torch.zeros(E, Lq, dtype=torch.long)
        q_live = torch.zeros(E, Lq, dtype=torch.bool)
        seg_of = torch.full((E, Lq), -1, dtype=torch.long)
        for e in range(E):
            n = len(q_rows[e])
            q_index[e, :n], q_live[e, :n], seg_of[e, :n] = q_rows[e], True, q_seg[e]
        causal = torch.arange(Lq)[:, None] >= torch.arange(Lq)[None]
        same = (seg_of[:, :, None] == seg_of[:, None, :]) & (seg_of[:, :, None] >= 0)
        self_mask = causal[None] & same
        self_mask |= torch.eye(Lq, dtype=torch.bool)[None] & ~q_live[:, :, None]  # 填充位自看
        cache_mask = (torch.arange(A)[None] < torch.tensor(cache_len)[:, None])[:, None, :].expand(E, Lq, A) \
            & q_live[:, :, None]
        self.mask = put(torch.cat([cache_mask, self_mask], 2)[:, None])
        self.q_index, self.q_live = put(q_index), put(q_live)
        self.A = A


class _Cache:
    """每层的上下文: 全注意力层的 K / V 流与每题的下标; DeltaNet 层的每题递归状态与卷积尾."""

    def __init__(self, E):
        self.E = E
        self.k, self.v = {}, {}         # 层号 -> (N, kv_heads, head_dim), 按追加顺序
        self.rows = [[] for _ in range(E)]  # 每题在 K / V 流里的行号 (前缀在前, 隐式位置依次)
        self.state, self.tail = {}, {}  # 层号 -> (E, H, K, V) fp32 / (E, w, C)

    def length(self):
        return [len(r) for r in self.rows]

    def index(self, device):
        A = max(1, max(self.length()))
        idx = torch.zeros(self.E, A, dtype=torch.long)
        for e, r in enumerate(self.rows):
            if r:
                idx[e, :len(r)] = torch.tensor(r)
        return idx.to(device)


def _attention(attn, x, cos, sin, stage, past_k, past_v, cache_index):
    """x: 本阶段的 (T, hidden). 返回 (输出 (T, hidden), 本阶段旋转后的 k, v (T, kv_heads, head_dim))."""
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
    if stage.A:
        keys = torch.cat([past_k[cache_index[:, :stage.A]], own_k], 1)
        vals = torch.cat([past_v[cache_index[:, :stage.A]], own_v], 1)
    else:
        keys, vals = own_k, own_v
    keys, vals = keys.transpose(1, 2), vals.transpose(1, 2)
    rep = rows_q.shape[1] // keys.shape[1]
    keys, vals = keys.repeat_interleave(rep, 1), vals.repeat_interleave(rep, 1)
    out = F.scaled_dot_product_attention(rows_q, keys, vals, attn_mask=stage.mask, scale=attn.scaling)
    out = out.transpose(1, 2)[stage.q_live].reshape(T, -1) * torch.sigmoid(gate)
    return attn.o_proj(out), k, v


def _deltanet(layer, x, stage, state, tail, want_final):
    """x: (T, hidden). state: (E, H, K, V) fp32 或 None (零); tail: (E, w, C) 或 None (零).
    返回 (输出, 每题末状态 (E, ...) 或 None, 每题新卷积尾 (E, w, C) 或 None)."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    T, E, w = x.shape[0], stage.E, stage.conv_width
    raw = layer.in_proj_qkv(x)
    ctx = tail if tail is not None else raw.new_zeros(E, w, raw.shape[1])
    weight = layer.conv1d.weight
    padded = torch.cat([raw, ctx.reshape(E * w, -1).to(raw.dtype)])
    stream = stage.conv_gather(padded).T[None].to(weight.dtype)
    conv = F.conv1d(stream, weight, layer.conv1d.bias, groups=stream.shape[1])[0].T
    # conv 输出第 j 行 = 输入第 j..j+w 列; 段内第 i 个 token 在输入里排第 w + i, 对应输出第 i 行
    qkv = F.silu(conv.index_select(0, stage.conv_keep - w)).to(raw.dtype)
    z = layer.in_proj_z(x).reshape(T, -1, layer.head_v_dim)
    beta = layer.in_proj_b(x).sigmoid()
    g = -layer.A_log.float().exp() * F.softplus(layer.in_proj_a(x).float() + layer.dt_bias)
    q, k, v = torch.split(qkv, [layer.key_dim, layer.key_dim, layer.value_dim], dim=-1)
    q, k, v = q.reshape(T, -1, layer.head_k_dim), k.reshape(T, -1, layer.head_k_dim), v.reshape(T, -1, layer.head_v_dim)
    if layer.num_v_heads // layer.num_k_heads > 1:
        rep = layer.num_v_heads // layer.num_k_heads
        q, k = q.repeat_interleave(rep, dim=1), k.repeat_interleave(rep, dim=1)
    init = state.index_select(0, stage.owner) if state is not None else None
    cu = stage.cu
    n = len(cu) - 1
    per_call = max(1, _packed.KERNEL_GRID_ROWS // v.shape[1])
    bounds = cu.tolist()
    outs, finals = [], []
    for s in range(0, n, per_call):
        e = min(n, s + per_call)
        a, b = bounds[s], bounds[e]
        o, final = chunk_gated_delta_rule(q[a:b][None], k[a:b][None], v[a:b][None], g=g[a:b][None], beta=beta[a:b][None],
                                          initial_state=None if init is None else init[s:e],
                                          output_final_state=True, use_qk_l2norm_in_kernel=True,
                                          cu_seqlens=cu[s:e + 1] - cu[s])
        outs.append(o[0])
        finals.append(final)
    out = torch.cat(outs)
    core = layer.norm(out.reshape(-1, layer.head_v_dim), z.reshape(-1, layer.head_v_dim))
    y = layer.out_proj(core.reshape(T, -1))
    if not want_final:
        return y, None, None
    seg = torch.tensor(stage.example_tail, device=x.device)
    final = torch.cat(finals).float().index_select(0, seg)
    # 新卷积尾: 每题 (旧尾 + 本阶段这题那段的原始输入) 的最后 w 列
    new_tail = []
    for e in range(E):
        s = stage.example_tail[e]
        st, en = stage.seg_start[s], (stage.seg_start[s + 1] if s + 1 < n else T)
        new_tail.append(torch.cat([ctx[e].to(raw.dtype), raw[st:en]])[-w:])
    return y, final, torch.stack(new_tail)


def _stage_layer(layer, h, cos, sin, stage, past_k, past_v, cache_index, state, tail, want_cache):
    residual = h
    x = layer.input_layernorm(h)
    k = v = final = new_tail = None
    if layer.block_type == "linear_attention":
        x, final, new_tail = _deltanet(layer.linear_attn, x, stage, state, tail, want_cache)
    else:
        x, k, v = _attention(layer.self_attn, x, cos, sin, stage, past_k, past_v, cache_index)
    h = residual + x
    h = h + layer.mlp(layer.post_attention_layernorm(h))
    return h, k, v, final, new_tail


def _run_stage(decoder, h, stage, cache, checkpointed, want_cache):
    """本阶段的全部层; want_cache 时把本阶段的 K / V、末状态、卷积尾记进 cache. 返回 final norm 之后的 h."""
    cos, sin = decoder.rotary_emb(h[None], stage.pos[:, None])
    cos, sin = cos[0], sin[0]
    cache_index = cache.index(h.device)
    for i, layer in enumerate(decoder.layers[:decoder.config.num_hidden_layers]):
        args = (layer, h, cos, sin, stage, cache.k.get(i), cache.v.get(i), cache_index,
                cache.state.get(i), cache.tail.get(i), want_cache)
        if checkpointed and torch.is_grad_enabled():
            h, k, v, final, tail = checkpoint(_stage_layer, *args, use_reentrant=False)
        else:
            h, k, v, final, tail = _stage_layer(*args)
        if want_cache:
            if k is not None:
                cache.k[i] = k if i not in cache.k else torch.cat([cache.k[i], k])
                cache.v[i] = v if i not in cache.v else torch.cat([cache.v[i], v])
            if final is not None:
                cache.state[i], cache.tail[i] = final, tail
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

    # 阶段 0: 前缀
    stage = _Stage(prefixes, list(range(E)), [0] * E, [0] * E, w, device)
    h = _run_stage(decoder, decoder.embed_tokens(stage.ids), stage, cache, checkpointed, True)
    for e in range(E):
        cache.rows[e] = list(range(stage.seg_start[e], stage.seg_start[e] + plen[e]))
    total = stage.T
    x = h.index_select(0, stage.last)  # (E, hidden): 第 1 个隐式位置的输入
    # 阶段 1..steps: 每题 1 个隐式位置
    for j in range(1, steps + 1):
        one = [torch.zeros(1, dtype=torch.long)] * E
        stage = _Stage(one, list(range(E)), [plen[e] + j - 1 for e in range(E)], cache.length(), w, device)
        h = _run_stage(decoder, x, stage, cache, checkpointed, True)
        for e in range(E):
            cache.rows[e].append(total + e)
        total += E
        x = h
    # 末阶段: 各选项分支, 从最后一个隐式位置之后接着数位置
    lengths = cpu["suffix_attention_mask"].sum(-1)
    index = {r: e for e, r in enumerate(examples)}
    segs, owner = [], []
    for r, c in rows:
        n = int(lengths[r, c])
        if n < 1:
            raise ValueError("candidate continuation needs a readout token")
        segs.append(cpu["suffix_input_ids"][r, c, -n:])
        owner.append(index[r])
    stage = _Stage(segs, owner, [plen[e] + steps for e in owner], cache.length(), w, device)
    h = _run_stage(decoder, decoder.embed_tokens(stage.ids), stage, cache, checkpointed, False)
    return h.index_select(0, stage.last)
