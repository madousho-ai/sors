"""Exact (float64, pure torch) gated delta rule for checking the packed path's gradients."""
import torch
import torch.nn.functional as F


def _one(q, k, v, g, beta, S):
    K = q.shape[-1]
    q = F.normalize(q, dim=-1, eps=1e-6) * K ** -0.5
    k = F.normalize(k, dim=-1, eps=1e-6)
    outs = []
    for t in range(q.shape[0]):
        S = S * g[t].exp()[:, None, None]
        kv = (S * k[t][:, :, None]).sum(-2)
        S = S + k[t][:, :, None] * ((v[t] - kv) * beta[t][:, None])[:, None, :]
        outs.append((S * q[t][:, :, None]).sum(-2))
    return torch.stack(outs), S


def exact_gated_delta_rule(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
                           use_qk_l2norm_in_kernel=True, cu_seqlens=None, **_):
    B, T, H, K = q.shape
    V = v.shape[-1]
    if cu_seqlens is None:
        bounds = [(b, 0, T) for b in range(B)]
    else:
        c = cu_seqlens.tolist()
        bounds = [(0, c[i], c[i + 1]) for i in range(len(c) - 1)]
    out = torch.zeros_like(v)
    finals = []
    for n, (b, s, e) in enumerate(bounds):
        S = (initial_state[n].to(q.dtype) if initial_state is not None else q.new_zeros(H, K, V))
        o, S = _one(q[b, s:e], k[b, s:e], v[b, s:e], g[b, s:e].to(q.dtype), beta[b, s:e].to(q.dtype), S)
        out = out.index_put((torch.full((e - s,), b, device=q.device), torch.arange(s, e, device=q.device)), o)
        finals.append(S)
    return out, (torch.stack(finals) if output_final_state else None)
