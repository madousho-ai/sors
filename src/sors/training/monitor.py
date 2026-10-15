"""训练监控: 各参数组离开训练起点多远.

分三组, 与 trainable_param_groups 的划分一致但不看 --lr-decision 给没给:
backbone (base. 下除 token 行以外), rows (token 行), decision (base 之外新增的决策层).
起点是这个进程开始训练时的权重 (续训时就是续训点). 主干的起点副本放 CPU:
2B 主干再存一份放不进显存; 它也只在评估点算. 小的两组副本留在参数所在设备上, 每个日志步都算.
"""

import torch


class ParamDrift:
    def __init__(self, m):
        decision = getattr(m, "decision_config", None) is not None
        self.params: dict[str, list[torch.nn.Parameter]] = {"backbone": [], "rows": []}
        if decision:
            self.params["decision"] = []
        for n, p in m.named_parameters():
            if not p.requires_grad:
                continue
            if n.endswith(".rows"):
                role = "rows"
            elif decision and not n.startswith("base."):
                role = "decision"
            else:
                role = "backbone"
            self.params[role].append(p)
        with torch.no_grad():
            self.start = {role: [p.detach().to("cpu", copy=True) if role == "backbone" else p.detach().clone()
                                 for p in ps] for role, ps in self.params.items()}

    @torch.no_grad()
    def ratios(self, roles=None) -> dict[str, float]:
        """||W - W0|| / ||W0||, 每组把全部张量当成一个向量."""
        out = {}
        for role in roles or self.params:
            moved = norm = 0.0
            for p, s in zip(self.params[role], self.start[role]):
                w = p.detach().to(s.device).float()
                s = s.float()
                moved += float((w - s).square().sum())
                norm += float(s.square().sum())
            out[role] = (moved / norm) ** 0.5 if norm > 0 else 0.0
        return out


def writeback_blocks(m) -> list:
    """开了回写的决策块, 打开它们的回写统计后按块序号返回."""
    blocks = [b for b in getattr(m, "blocks", []) if getattr(b, "write_out", None) is not None]
    for b in blocks:
        b.track_writeback(True)
    return blocks
