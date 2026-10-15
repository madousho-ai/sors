"""训练中的两样监控: 各参数组离开起点多远 (drift), 决策块回写进主干的幅度 (writeback)."""

import torch

from test_decision import block, examples, inputs, tiny_model


def test_writeback_ratio_is_the_feedback_size_relative_to_the_text_stream_it_writes_into():
    """只在 train 模式下记; 读一次清零; 零初始化的回写是 0; 没记过是 None."""
    m = block(True).train()
    m.track_writeback(True)
    u, h, om, hm = inputs()
    m(u, h, om, hm)
    assert m.pop_writeback_ratio() == 0.0, "the zero-initialized write path already writes"
    assert m.pop_writeback_ratio() is None, "reading the ratio did not reset it"

    with torch.no_grad():
        m.write_out.weight.normal_(std=0.1)
    _, got = m(u, h, om, hm)
    _, again = m(u, h, om, hm)
    delta = torch.cat([got - h, again - h])
    text = torch.cat([h[hm], h[hm]])
    expected = (delta.norm() / text.norm()).item()
    assert abs(m.pop_writeback_ratio() - expected) < 1e-5

    m.eval()
    m(u, h, om, hm)
    assert m.pop_writeback_ratio() is None, "eval forwards leaked into the training statistic"


def test_writeback_is_not_tracked_unless_asked():
    m = block(True).train()
    m(*inputs())
    assert m.pop_writeback_ratio() is None


def test_drift_reports_each_role_relative_to_its_starting_weights():
    """主干 / token 行 / 决策层分开算 ||W - W0|| / ||W0||, 与 --lr-decision 给没给无关."""
    from sors.training.monitor import ParamDrift

    m, *_ = tiny_model("minimal", True, trainable="full")
    drift = ParamDrift(m)
    assert drift.ratios() == {"backbone": 0.0, "rows": 0.0, "decision": 0.0}

    p = m.option_proj.weight
    with torch.no_grad():
        before = p.detach().clone()
        p.add_(0.5)
    decision = [q for n, q in m.named_parameters() if q.requires_grad and not n.startswith("base.")]
    start = torch.cat([before.flatten()] + [q.detach().flatten() for q in decision if q is not p])
    expected = ((p - before).norm() / start.norm()).item()
    got = drift.ratios()
    assert got["backbone"] == 0.0 and got["rows"] == 0.0
    assert abs(got["decision"] - expected) < 1e-6, (got, expected)


def test_drift_has_no_decision_role_without_decision_layers():
    from sors.core.model import prepare_model
    from sors.training.monitor import ParamDrift
    from test_decision import tiny_backbone, tokenizer

    tok, d, ids = tokenizer()
    m = prepare_model(tiny_backbone(tok), ids, 4, 8, 0.0, trainable="full")
    assert set(ParamDrift(m).ratios()) == {"backbone", "rows"}


class _Writer:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def flush(self):
        pass


def test_backbone_drift_keeps_its_starting_copy_off_the_accelerator():
    """主干的起点副本放 CPU (2B 主干的副本放不进显存); 小的两组留在参数所在的设备上."""
    from sors.training.monitor import ParamDrift

    m, *_ = tiny_model("minimal", True, trainable="full")
    drift = ParamDrift(m)
    assert all(t.device.type == "cpu" for t in drift.start["backbone"])
    assert set(drift.ratios(["decision", "rows"])) == {"decision", "rows"}


def test_training_logs_small_drift_and_writeback_every_log_step_and_backbone_drift_at_eval_points():
    from sors.training.loop import TrainConfig, train

    m, tok, d, _ = tiny_model("minimal", True, trainable="full")
    w = _Writer()
    hist = train(m, tok, d, lambda n, rng: examples()[:n], {},
                 TrainConfig(steps=2, batch_size=2, k_max=4, loss="menu", eval_every=2, log_every=1,
                             lr_lora=1e-3, lr_embed=1e-3, lr_decision=1e-2), writer=w)
    tags = {(t, s) for t, _, s in w.scalars}
    for step in (1, 2):
        for role in ("rows", "decision"):
            assert (f"train/drift/{role}", step) in tags, (role, step, sorted(tags))
        for j in (0, 1):
            assert (f"train/writeback/block{j}", step) in tags, (j, step)
    assert [s for t, _, s in w.scalars if t == "train/drift/backbone"] == [0, 2]
    decision = [v for t, v, s in w.scalars if t == "train/drift/decision" and s > 0]
    assert 0 < decision[0] < decision[1], decision
    assert [set(r["drift"]) for r in hist] == [{"backbone", "rows", "decision"}] * 2
    assert hist[0]["drift"]["backbone"] == 0.0 < hist[1]["drift"]["backbone"]
