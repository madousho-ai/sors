"""Explicit CUDA check with tiny real backbones; no model downloads or full training."""

import json
from pathlib import Path
import random
import tempfile

import pytest
import torch

from test_decision import batch, examples, tiny_backbone, tiny_model
from sors.core.checkpoint import prepare_from_checkpoint, save_trained
from sors.core.menu import reorder_menu, with_partners
from sors.training.loop import TrainConfig
from sors.training.loss import training_loss


def check(family, kind, feedback):
    print(f"START CUDA {family} {kind} feedback={feedback}", flush=True)
    m, tok, d, train_ids = tiny_model(kind, feedback, family=family, trainable="full")
    m.base.to(device="cuda", dtype=torch.bfloat16)
    m.to(device="cuda")
    with torch.no_grad():
        if feedback:
            for block in m.blocks:
                block.write_out.weight.normal_(std=0.02)
    pairs = with_partners(examples(), random.Random(0))
    data = {k: v.cuda() for k, v in batch(m, tok, d, pairs).items()}
    gradients, outputs = [], []
    for checkpointed in (False, True):
        if checkpointed:
            rebuilt, _, _, _ = tiny_model(kind, feedback, grad_ckpt=True, family=family, trainable="full")
            rebuilt.base.to(device="cuda", dtype=torch.bfloat16)
            rebuilt.to(device="cuda")
            rebuilt.load_state_dict(m.state_dict())
            m = rebuilt
        m.train()
        m.zero_grad(set_to_none=True)
        # The original coupled forward is the read-only architecture equivalence oracle.
        logits = m._forward_batch(data) if not feedback and not checkpointed else m.forward_batch(data)
        outputs.append(logits.detach().cpu())
        loss = training_loss("menu", logits, data["slot_ids"], data["gold"], d, data["target"])
        loss.backward()
        assert torch.isfinite(loss)
        grads = {n: p.grad.detach().float().cpu().clone() for n, p in m.named_parameters() if p.grad is not None}
        assert all(torch.isfinite(g).all() for g in grads.values())
        assert grads["blocks.0.read.in_proj_weight"].abs().sum() > 0
        if feedback:
            assert all(grads[f"blocks.{i}.write_out.weight"].abs().sum() > 0 for i in range(len(m.blocks)))
        gradients.append(grads)
    torch.testing.assert_close(outputs[0], outputs[1], atol=3e-6, rtol=3e-5)
    assert gradients[0].keys() == gradients[1].keys()
    for name, grad in gradients[0].items():
        torch.testing.assert_close(grad, gradients[1][name], atol=3e-3, rtol=3e-2, msg=name)
    m.eval()
    with torch.no_grad():
        want = m.forward_batch(data)
        symmetry_error = None
        if kind == "structural":
            ex = examples()[0]
            permuted = reorder_menu(ex, [2, 0, 1], [200, 90, 155])
            a, b = ({k: v.cuda() for k, v in batch(m, tok, d, [item]).items()} for item in (ex, permuted))
            pa = m.forward_batch(a)[0, a["slot_ids"][0, :3]].softmax(-1)
            pb = m.forward_batch(b)[0, b["slot_ids"][0, :3]].softmax(-1)
            symmetry_error = float((pb - pa[[2, 0, 1]]).abs().max())
            torch.testing.assert_close(pb, pa[[2, 0, 1]], atol=3e-3, rtol=3e-2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.safetensors"
            save_trained(m, train_ids, TrainConfig(loss="menu"), path)
            raw = tiny_backbone(tok, family).to(device="cuda", dtype=torch.bfloat16)
            restored, _ = prepare_from_checkpoint(raw, train_ids, path)
            torch.testing.assert_close(restored.eval().forward_batch(data), want, atol=0, rtol=0)
    print(json.dumps({"family": family, "architecture": kind, "feedback": feedback,
                      "loss": float(loss.detach()), "symmetry_max_abs": symmetry_error}), flush=True)


@pytest.fixture(autouse=True)
def _memory_fraction(require_cuda):
    torch.cuda.set_per_process_memory_fraction(0.1)
    yield
    torch.cuda.set_per_process_memory_fraction(1.0)  # 同一进程里接着跑的测试不受这个上限约束


@pytest.mark.parametrize("feedback", [False, True])
@pytest.mark.parametrize("kind", ["minimal", "structural"])
@pytest.mark.parametrize("family", ["qwen3", "qwen35"])
def test_cuda(family, kind, feedback):
    check(family, kind, feedback)
