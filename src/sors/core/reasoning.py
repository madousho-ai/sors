"""推理内存模块: 每道题一个 delta-rule 矩阵 M, 草稿位在上面读写几轮, 选项读出 token 再读 M.

1. 写入: 前缀 (材料 + 问题) 的每个 token 按 GDN 的 delta rule 写进 M (每头 head_dim x head_dim, 不衰减).
   与主干的 Gated DeltaNet 用同一个 kernel (transformers 的 torch_chunk_gated_delta_rule; 装了 fla 就是 fla 的).
2. 推理 steps 轮: slots 个草稿位各用自己的状态读 M, 再交叉注意力回看本题前缀, 两者合起来过 MLP 更新自己,
   然后按 delta rule 把自己写回 M. 下一轮在更新过的 M 上接着算.
3. 读出: 每个选项的读出向量读 M, 经 out 投影成一个加在读出向量上的增量. out 初始为零, 刚接上时增量恰好为零,
   读到的内容非零, 所以 out 第一步就有梯度.

只看每题自己的前缀; 一批里的题互不相干. steps=0 时草稿位不参与, 只剩写入与读出.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _delta_write(state, k, v, beta):
    """state (N, H, K, V); k (N, H, K); v (N, H, V); beta (N, H). 与 GDN 相同: k 先 L2 归一化."""
    k = F.normalize(k, dim=-1, eps=1e-6)
    old = (state * k[..., None]).sum(-2)
    return state + k[..., None] * ((v - old) * beta[..., None])[..., None, :]


def _read(state, q):
    """q (N, H, K) -> (N, H, V); q 与 GDN 一样 L2 归一化后乘 K^-1/2."""
    q = F.normalize(q, dim=-1, eps=1e-6) * q.shape[-1] ** -0.5
    return (state * q[..., None]).sum(-2)


class ReasonMemory(nn.Module):
    def __init__(self, hidden: int, heads: int = 8, head_dim: int = 128, slots: int = 8, slot_dim: int = 512,
                 steps: int = 3, attn_heads: int = 8):
        super().__init__()
        if slot_dim % attn_heads:
            raise ValueError("slot_dim must be divisible by attn_heads")
        self.heads, self.head_dim, self.steps, self.attn_heads = heads, head_dim, steps, attn_heads
        mem = heads * head_dim
        self.norm = nn.RMSNorm(hidden, eps=1e-6)
        self.write_kv = nn.Linear(hidden, 2 * mem, bias=False)
        self.write_beta = nn.Linear(hidden, heads)
        self.context_kv = nn.Linear(hidden, 2 * slot_dim, bias=False)
        self.slots = nn.Parameter(torch.randn(slots, slot_dim) * 0.02)
        self.slot_norm = nn.RMSNorm(slot_dim, eps=1e-6)
        self.slot_query = nn.Linear(slot_dim, mem + slot_dim, bias=False)  # 读 M 的 q 与回看前缀的 q
        self.mix_norm = nn.RMSNorm(mem + slot_dim, eps=1e-6)
        self.step = nn.Sequential(nn.Linear(mem + slot_dim, 4 * slot_dim), nn.GELU(), nn.Linear(4 * slot_dim, slot_dim))
        self.slot_write = nn.Linear(slot_dim, 2 * mem + heads, bias=False)
        self.read_norm = nn.RMSNorm(hidden, eps=1e-6)
        self.read_query = nn.Linear(hidden, mem, bias=False)
        self.out = nn.Linear(mem, hidden, bias=False)
        nn.init.zeros_(self.out.weight)

    def _memory(self, x, lengths):
        """x: 各题前缀拼成的 (sum P, hidden). 返回每题写完前缀后的 M, (N, H, K, V), float32."""
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen35

        H, D = self.heads, self.head_dim
        k, v = self.write_kv(x).view(-1, 2, H, D).unbind(1)
        beta = torch.sigmoid(self.write_beta(x))
        cu = torch.tensor([0, *lengths], device=x.device).cumsum(0)
        _, state = qwen35.torch_chunk_gated_delta_rule(
            k[None], k[None], v[None], g=beta.new_zeros(beta.shape)[None].float(), beta=beta[None],
            initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
        return state.to(x.dtype)

    def forward(self, prefixes: list[torch.Tensor], readout: torch.Tensor, owner: torch.Tensor) -> torch.Tensor:
        """prefixes: 每题一个 (P_i, hidden); readout: (R, hidden) 各选项的读出向量; owner: (R,) 每个选项属于第几题.
        返回加在 readout 上的增量 (R, hidden)."""
        lengths = [len(p) for p in prefixes]
        x = self.norm(torch.cat(prefixes).to(self.out.weight.dtype))
        state = self._memory(x, lengths)
        N, H, D, mem = len(prefixes), self.heads, self.head_dim, self.heads * self.head_dim
        if self.steps:
            width = max(lengths)
            padded = x.new_zeros(N, width, x.shape[-1])
            live = torch.zeros(N, width, dtype=torch.bool, device=x.device)
            for i, (p, n) in enumerate(zip(x.split(lengths), lengths)):
                padded[i, :n], live[i, :n] = p, True
            a, dh = self.attn_heads, self.slots.shape[1] // self.attn_heads
            ck, cv = self.context_kv(padded).view(N, width, 2, a, dh).permute(2, 0, 3, 1, 4)
            mask = live[:, None, None, :]
            s = self.slots[None].expand(N, -1, -1)
            for _ in range(self.steps):
                q_mem, q_ctx = self.slot_query(self.slot_norm(s)).split([mem, s.shape[-1]], -1)
                got = _read(state[:, None], q_mem.view(N, -1, H, D)).flatten(2)  # (N, S, mem)
                q_ctx = q_ctx.view(N, -1, a, dh).transpose(1, 2)
                ctx = F.scaled_dot_product_attention(q_ctx, ck, cv, attn_mask=mask).transpose(1, 2).flatten(2)
                s = s + self.step(self.mix_norm(torch.cat([got, ctx], -1)))
                k, v, beta = self.slot_write(s).split([mem, mem, H], -1)
                k, v, beta = k.view(N, -1, H, D), v.view(N, -1, H, D), torch.sigmoid(beta)
                for j in range(s.shape[1]):  # 草稿位按顺序写回, 与前缀写入同一条规则
                    state = _delta_write(state, k[:, j], v[:, j], beta[:, j])
        q = self.read_query(self.read_norm(readout.to(self.out.weight.dtype))).view(-1, H, D)
        return self.out(_read(state[owner], q).flatten(1))
