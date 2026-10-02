"""Minimal read-only decisions retain their math with layer-local recomputation."""

import random
import threading
from dataclasses import replace
from unittest.mock import patch

import torch
from torch.utils.checkpoint import set_checkpoint_early_stop

from _runner import run
from test_decision import batch, examples, tiny_model
from decidophobia.core.menu import with_partners
from decidophobia.core.model import decision_logits, trainable_param_groups
from decidophobia.training.loop import TrainConfig, step_loss


def test_minimal_backward_recomputes_layers_individually_and_decision_blocks_separately():
    for family in ("qwen3", "qwen35"):
        m, tok, d, _ = tiny_model("minimal", False, grad_ckpt=True, family=family, trainable="full")
        m.train()
        events, handles = [], []
        for i, layer in enumerate(m.decoder.layers):
            handles.append(layer.register_forward_pre_hook(lambda _m, _args, i=i: events.append(("text", i))))
        for i, layer in enumerate(m.blocks):
            handles.append(layer.register_forward_pre_hook(lambda _m, _args, i=i: events.append(("decision", i))))
        try:
            data = batch(m, tok, d)
            logits = m.forward_batch(data)
            events.clear()
            loss = -logits.log_softmax(-1)[:, d[0]].mean()
            loss.backward()
            text = [i for kind, i in events if kind == "text"]
            decisions = [i for kind, i in events if kind == "decision"]
            assert text == list(reversed(range(len(m.decoder.layers)))), (family, events)
            assert decisions == [1, 0], (family, events)
        finally:
            for handle in handles:
                handle.remove()


def test_minimal_read_only_decisions_run_after_the_backbone():
    m, tok, d, _ = tiny_model("minimal", False, grad_ckpt=True, trainable="full")
    events = []
    handles = [m.decoder.register_forward_hook(lambda *_: events.append("text_done"))]
    for i, layer in enumerate(m.blocks):
        handles.append(layer.register_forward_pre_hook(lambda _m, _args, i=i: events.append(f"decision_{i}")))
    try:
        m.train().forward_batch(batch(m, tok, d))
        assert events == ["text_done", "decision_0", "decision_1"], events
    finally:
        for handle in handles:
            handle.remove()


def test_minimal_matches_coupled_path_logits_all_gradients_and_optimizer_updates():
    pairs = with_partners(examples(), random.Random(0))
    cfg = TrainConfig(loss="menu", consistency=1.0)
    for family in ("qwen3", "qwen35"):
        for trainable in ("full", "decision-only"):
            for checkpointed, groups in ((False, 1), (True, 1), (True, 2)):
                reference, tok, d, _ = tiny_model("minimal", False, family=family, trainable=trainable)
                actual, _, _, _ = tiny_model("minimal", False, grad_ckpt=checkpointed,
                                            family=family, trainable=trainable)
                # Make substituting the final normalized state for a raw layer output observable.
                with torch.no_grad():
                    reference.decoder.norm.weight.copy_(torch.linspace(0.4, 1.7, reference.config.hidden_size))
                actual.load_state_dict(reference.state_dict())
                data = batch(reference, tok, d, pairs)
                optimizers, logits = [], []
                for m, legacy in ((reference, True), (actual, False)):
                    m.train()
                    optimizer = torch.optim.SGD(trainable_param_groups(m, 1e-5, 1e-3))
                    optimizer.zero_grad(set_to_none=True)
                    # The original coupled forward remains the reference for feedback-enabled models.
                    z = m._forward_batch(data) if legacy else decision_logits(m, data, groups)
                    loss, _, _ = step_loss(cfg, pairs, data, z, d)
                    loss.backward()
                    assert torch.isfinite(loss)
                    assert m.blocks[0].read.in_proj_weight.grad.abs().sum() > 0
                    assert m.get_input_embeddings().rows.grad.abs().sum() > 0
                    optimizers.append(optimizer)
                    logits.append(z.detach())
                torch.testing.assert_close(logits[0], logits[1], atol=3e-6, rtol=3e-5)
                for (name, p), (other, q) in zip(reference.named_parameters(), actual.named_parameters()):
                    assert name == other
                    assert (p.grad is None) == (q.grad is None), name
                    if p.grad is not None:
                        assert torch.isfinite(q.grad).all(), name
                        torch.testing.assert_close(p.grad, q.grad, atol=3e-5, rtol=3e-4, msg=name)
                for optimizer in optimizers:
                    optimizer.step()
                for (name, p), (_, q) in zip(reference.named_parameters(), actual.named_parameters()):
                    torch.testing.assert_close(p, q, atol=3e-6, rtol=3e-5, msg=name)
                assert all(not layer._forward_hooks for layer in actual.decoder.layers)


def test_minimal_capture_cleans_up_after_failure_and_preserves_foreign_hooks():
    m, tok, d, _ = tiny_model("minimal", False, grad_ckpt=True, trainable="full")
    data = batch(m, tok, d)
    calls = []
    foreign = m.decoder.layers[0].register_forward_hook(lambda *_: calls.append(1))
    try:
        want = m.eval().forward_batch(data).detach()
        with patch.object(m.decoder.layers[-1], "forward", side_effect=RuntimeError("injected encoder failure")):
            try:
                m.train().forward_batch(data)
            except RuntimeError as exc:
                assert str(exc) == "injected encoder failure"
            else:
                raise AssertionError("fault injection did not reach the text encoder")
        assert list(m.decoder.layers[0]._forward_hooks) == [foreign.id]
        assert all(not layer._forward_hooks for layer in m.decoder.layers[1:])
        torch.testing.assert_close(m.eval().forward_batch(data), want, atol=0, rtol=0)
        assert len(calls) == 3
    finally:
        foreign.remove()


def test_minimal_capture_ignores_another_threads_complete_backward_replay():
    for family in ("qwen3", "qwen35"):
        m, tok, d, _ = tiny_model("minimal", False, grad_ckpt=True, family=family, trainable="full")
        a = batch(m, tok, d)
        changed = [replace(ex, query="green green short", option_names=list(reversed(ex.option_names)))
                   for ex in examples()]
        b = batch(m, tok, d, changed)
        with torch.no_grad():
            expected = m.eval().forward_batch(b)
        m.train()
        with set_checkpoint_early_stop(False):
            pending = -m.forward_batch(a).log_softmax(-1)[:, d[0]].mean()
        reached, release = threading.Event(), threading.Event()
        result, errors = [], []

        def pause_at_final_norm(_module, _args):
            reached.set()
            assert release.wait(10), "concurrent-forward test was not released"

        def forward_b():
            try:
                result.append(m.forward_batch(b).detach())
            except Exception as exc:
                errors.append(exc)

        handle = m.decoder.norm.register_forward_pre_hook(pause_at_final_norm)
        worker = threading.Thread(target=forward_b, daemon=True)
        try:
            worker.start()
            assert reached.wait(10), errors
            pending.backward()
        finally:
            release.set()
            worker.join(10)
            handle.remove()
        assert not worker.is_alive() and not errors, errors
        torch.testing.assert_close(result[0], expected, atol=3e-6, rtol=3e-5)


if __name__ == "__main__":
    run(globals())
