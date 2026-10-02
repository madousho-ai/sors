"""Explicit CUDA matrix for candidate joint encoding; tiny models, no downloads."""

import json
from pathlib import Path
import random
import tempfile

import torch

from _runner import run
from test_candidate import candidate_batch, candidate_model, examples, tiny_backbone
from sors.core.checkpoint import prepare_from_checkpoint, save_trained
from sors.core.menu import reorder_menu, with_partners
from sors.training.loop import TrainConfig
from sors.training.loss import training_loss


def check(family, blocks):
    print(f"START CUDA candidate {family} blocks={blocks}", flush=True)
    m, tok, d, train_ids = candidate_model(blocks, family, trainable="full")
    m.base.to(device="cuda", dtype=torch.bfloat16)
    m.to(device="cuda")
    exs = with_partners(examples(), random.Random(0))
    data = {k: v.cuda() for k, v in candidate_batch(tok, d, exs).items()}
    gradients = []
    for checkpointed in (False, True):
        m.checkpoint_forward = checkpointed
        m.train()
        m.zero_grad(set_to_none=True)
        logits = m.forward_batch(data)
        loss = training_loss("menu", logits, data["slot_ids"], data["gold"], d, data["target"])
        loss.backward()
        grads = {n: p.grad.detach().float().cpu().clone() for n, p in m.named_parameters() if p.grad is not None}
        assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in grads.values())
        assert grads["scorer.weight"].abs().sum() > 0
        if blocks:
            assert grads["blocks.0.attention.in_proj_weight"].abs().sum() > 0
        gradients.append(grads)
    assert gradients[0].keys() == gradients[1].keys()
    for name, grad in gradients[0].items():
        torch.testing.assert_close(grad, gradients[1][name], atol=3e-3, rtol=3e-2, msg=name)
    m.eval()
    with torch.no_grad():
        want = m.forward_batch(data)
        ex = examples()[0]
        changed = reorder_menu(ex, [2, 0, 1], [200, 7, 111])
        a, b = ({k: v.cuda() for k, v in candidate_batch(tok, d, [item]).items()} for item in (ex, changed))
        pa = m.forward_batch(a)[0, a["slot_ids"][0, :3]].softmax(-1)
        pb = m.forward_batch(b)[0, b["slot_ids"][0, :3]].softmax(-1)
        error = float((pb - pa[[2, 0, 1]]).abs().max())
        torch.testing.assert_close(pb, pa[[2, 0, 1]], atol=3e-3, rtol=3e-2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.safetensors"
            save_trained(m, train_ids, TrainConfig(loss="menu"), path)
            raw = tiny_backbone(tok, family).to(device="cuda", dtype=torch.bfloat16)
            restored, _ = prepare_from_checkpoint(raw, train_ids, path)
            torch.testing.assert_close(restored.eval().forward_batch(data), want, atol=0, rtol=0)
    print(json.dumps({"family": family, "blocks": blocks, "loss": float(loss.detach()),
                      "symmetry_max_abs": error}), flush=True)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; run this explicit validation on the GPU server")
    torch.cuda.set_per_process_memory_fraction(0.1)
    run({f"test_cuda_candidate_{family}_blocks_{blocks}":
         (lambda family=family, blocks=blocks: check(family, blocks))
         for family in ("qwen3", "qwen35") for blocks in (0, 1, 2)})
