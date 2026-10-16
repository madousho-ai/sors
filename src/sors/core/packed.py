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


class _Gather(torch.autograd.Function):
    """x.index_select(0, idx), 反传不用原子加. index_select 的反传是 index_add: 下标重复得多时
    (一题 256 个选项 = 512 个分支读同一份前缀末状态, 每份 16 x 128 x 128) 几百个线程原子加同一地址,
    一层就要 ~36 ms, 占整步 GPU 时间四成. 这里把梯度按目标行排好序, 每个目标行连续求和 (fp32),
    再按不重复的目标行写回."""

    @staticmethod
    def forward(ctx, x, idx, order, lengths, targets):
        ctx.rows = x.shape[0]
        ctx.save_for_backward(order, lengths, targets)
        return x.index_select(0, idx)

    @staticmethod
    def backward(ctx, grad):
        order, lengths, targets = ctx.saved_tensors
        flat = grad.index_select(0, order).flatten(1)
        sums = torch.segment_reduce(flat.float() if flat.dtype != torch.float64 else flat, "sum", lengths=lengths)
        out = grad.new_zeros((ctx.rows, flat.shape[1]))
        out.index_copy_(0, targets, sums.to(grad.dtype))
        return out.view(ctx.rows, *grad.shape[1:]), None, None, None, None


class _Index:
    """一份 gather 下标及其反传用的排序 (CPU 上算好)."""

    def __init__(self, idx: torch.Tensor, put):
        order = torch.argsort(idx, stable=True)
        targets, lengths = torch.unique_consecutive(idx[order], return_counts=True)
        self.idx, self.order, self.lengths, self.targets = put(idx), put(order), put(lengths), put(targets)

    def __call__(self, x):
        return _Gather.apply(x, self.idx, self.order, self.lengths, self.targets)


class _Plan:
    """流的布局. 段的顺序: 每个例子先前缀, 再各个分支 (按 valid.nonzero() 的顺序).

    全部在 CPU 上算好 (批里的小张量一次拷回), 每层要用的下标都是现成的设备张量:
    逐段切片在反传时每段都要分配一份整流大小的零张量, 段一多 (意图题 77~256 个分支) 就成了平方级开销.
    """

    def __init__(self, batch, valid, device, conv_width=3):
        cpu = {k: batch[k].cpu() for k in ("input_ids", "attention_mask", "prefix_lengths",
                                            "suffix_input_ids", "suffix_attention_mask")}
        valid_cpu = valid.cpu()
        rows = [tuple(x) for x in valid_cpu.nonzero().tolist()]
        by_row: dict[int, list[int]] = {}
        for r, c in rows:
            by_row.setdefault(r, []).append(c)
        tokens, pos = [], []
        self.seg_len, self.seg_kind, self.branch_prefix = [], [], []
        self.example_len = []
        readout = {}
        start = 0
        for r in sorted(by_row):
            prefix = cpu["input_ids"][r][cpu["attention_mask"][r].bool()][:int(cpu["prefix_lengths"][r])]
            p = len(prefix)
            ex_start, prefix_seg = start, -1
            if p:
                tokens.append(prefix)
                pos.append(torch.arange(p))
                prefix_seg = len(self.seg_len)
                self.seg_len.append(p), self.seg_kind.append(0), self.branch_prefix.append(-1)
                start += p
            lengths = cpu["suffix_attention_mask"][r].sum(-1).tolist()
            for c in by_row[r]:
                n = int(lengths[c])
                if n < 1:
                    raise ValueError("candidate continuation needs a readout token")
                tokens.append(cpu["suffix_input_ids"][r, c, -n:])
                pos.append(torch.arange(p, p + n))
                self.seg_len.append(n), self.seg_kind.append(1), self.branch_prefix.append(prefix_seg)
                start += n
                readout[(r, c)] = start - 1
            self.example_len.append(start - ex_start)
        T = start
        lens = torch.tensor(self.seg_len)
        self.seg_start = torch.cat([lens.new_zeros(1), lens.cumsum(0)[:-1]]).tolist()
        self.prefix_segs = [i for i, k in enumerate(self.seg_kind) if k == 0]
        self.branch_segs = [i for i, k in enumerate(self.seg_kind) if k == 1]

        def span(segs):
            return torch.cat([torch.arange(self.seg_start[s], self.seg_start[s] + self.seg_len[s]) for s in segs]) \
                if segs else torch.zeros(0, dtype=torch.long)

        def cu(segs):
            return torch.tensor([0] + [self.seg_len[s] for s in segs]).cumsum(0)

        # 短卷积: 每段前接 conv_width 列上下文; -1 = 零 (指向追加在末尾的零行)
        conv_index, conv_keep, at = [], [], 0
        for s, (st, n) in enumerate(zip(self.seg_start, self.seg_len)):
            p = self.branch_prefix[s]
            ctx = [T] * conv_width
            if p >= 0:
                end = self.seg_start[p] + self.seg_len[p]
                tail = list(range(max(self.seg_start[p], end - conv_width), end))
                ctx = [T] * (conv_width - len(tail)) + tail
            conv_index += ctx + list(range(st, st + n))
            conv_keep += range(at, at + n)  # conv 输出第 j 行对应 stream 第 j + width 列
            at += conv_width + n
        prefix_slot = {sg: i for i, sg in enumerate(self.prefix_segs)}
        branch_init = [prefix_slot.get(self.branch_prefix[s], len(self.prefix_segs)) for s in self.branch_segs]

        put = lambda t: t.to(device, non_blocking=True)
        self.ids = put(torch.cat(tokens))
        self.pos = put(torch.cat(pos))
        self.readout = put(torch.tensor([readout[k] for k in rows]))
        self.seg_id = put(torch.repeat_interleave(torch.arange(len(self.seg_len)), lens))
        self.conv_gather = _Index(torch.tensor(conv_index), put)
        self.conv_keep = put(torch.tensor(conv_keep))
        self.prefix_idx, self.prefix_cu = put(span(self.prefix_segs)), put(cu(self.prefix_segs))
        self.branch_idx, self.branch_cu = put(span(self.branch_segs)), put(cu(self.branch_segs))
        self.branch_gather = _Index(torch.tensor(branch_init, dtype=torch.long), put)
        self.conv_width = conv_width
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
    """每段的因果短卷积 + SiLU, 段前接上下文 (前缀接零, 分支接本例前缀末尾). qkv: (T, C).
    一次 gather 拼出带上下文的流, 一次卷积, 一次 gather 取回; 反传只有两次 index_add."""
    if layer.conv_kernel_size - 1 != plan.conv_width:
        raise ValueError("packed plan was built for a different conv kernel size")
    weight = layer.conv1d.weight  # (C, 1, k)
    padded = torch.cat([qkv, qkv.new_zeros(1, qkv.shape[1])])
    stream = plan.conv_gather(padded).T[None].to(weight.dtype)
    out = F.conv1d(stream, weight, layer.conv1d.bias, groups=stream.shape[1])[0].T
    return F.silu(out.index_select(0, plan.conv_keep)).to(qkv.dtype)


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

    def run(idx, cu, init):
        pick = lambda t: t.index_select(0, idx)[None]
        o, final = chunk_gated_delta_rule(pick(q), pick(k), pick(v), g=pick(g), beta=pick(beta), initial_state=init,
                                          output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
        return o[0], final

    out = v.new_zeros(v.shape)
    shape = (v.shape[1], q.shape[-1], v.shape[-1])
    finals = v.new_zeros((1, *shape), dtype=torch.float32)
    if plan.prefix_segs:
        o, final = run(plan.prefix_idx, plan.prefix_cu, None)
        out = out.index_copy(0, plan.prefix_idx, o.to(out.dtype))
        finals = torch.cat([final.float(), finals])  # 最后一行是零状态, 给没有前缀的分支
    if plan.branch_segs:
        o, _ = run(plan.branch_idx, plan.branch_cu, plan.branch_gather(finals))
        out = out.index_copy(0, plan.branch_idx, o.to(out.dtype))
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
    linear = [l.linear_attn for l in decoder.layers if getattr(l, "block_type", "") == "linear_attention"]
    plan = _Plan(batch, valid, batch["input_ids"].device, linear[0].conv_kernel_size - 1 if linear else 3)
    h = decoder.embed_tokens(plan.ids)  # (T, hidden)
    cos, sin = decoder.rotary_emb(h[None], plan.pos[None, None].expand(3, 1, -1))
    cos, sin = cos[0], sin[0]  # (T, rotary dim)
    for layer in decoder.layers[:decoder.config.num_hidden_layers]:
        if checkpointed and torch.is_grad_enabled():
            h = checkpoint(_layer, layer, h, cos, sin, plan, use_reentrant=False)
        else:
            h = _layer(layer, h, cos, sin, plan)
    return decoder.norm(h)[plan.readout]
