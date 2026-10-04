"""Feedback replays bounded text layers with explicit candidate-state gradients."""

import random
import copy
import pickle
import threading
from dataclasses import replace
from unittest.mock import patch

import torch
from torch.utils.checkpoint import set_checkpoint_early_stop

from _runner import run
from test_decision import batch, examples, tiny_model
from sors.core.menu import with_partners
from sors.core.model import decision_logits, trainable_param_groups
from sors.training.loop import TrainConfig, step_loss


def test_feedback_backward_replays_one_layer_at_a_time_in_reverse_order():
    # A whole-encoder checkpoint replays 0..depth, retaining all layers together.
    for family in ('qwen3', 'qwen35'):
        m, tok, d, _ = tiny_model('minimal', True, grad_ckpt=True, family=family, trainable='full')
        events, handles = [], []
        for i, layer in enumerate(m.decoder.layers):
            handles.append(layer.register_forward_pre_hook(lambda _m, _args, i=i: events.append(('text', i))))
        for i, layer in enumerate(m.blocks):
            handles.append(layer.register_forward_pre_hook(lambda _m, _args, i=i: events.append(('decision', i))))
        try:
            m.train()
            logits = m.forward_batch(batch(m, tok, d))
            events.clear()
            (-logits.log_softmax(-1)[:, d[0]].mean()).backward()
            assert [i for kind, i in events if kind == 'text'] == list(reversed(range(len(m.decoder.layers)))), events
            assert [i for kind, i in events if kind == 'decision'] == [1, 0], events
        finally:
            for handle in handles:
                handle.remove()


def test_feedback_logits_gradients_and_updates_match_original_coupled_forward():
    # Learned feedback, soft labels, JS partners, and multiple pending forwards
    # expose detached/reused candidate states and hidden-state normalization errors.
    pairs = with_partners(examples(), random.Random(0))
    cfg = TrainConfig(loss='menu', consistency=1.0)
    for family in ('qwen3', 'qwen35'):
        for trainable in ('full', 'decision-only'):
            for groups, early_stop in ((1, True), (2, True), (2, False)):
                reference, tok, d, _ = tiny_model('minimal', True, family=family, trainable=trainable)
                actual, _, _, _ = tiny_model('minimal', True, grad_ckpt=True, family=family, trainable=trainable)
                with torch.no_grad():
                    for block in reference.blocks:
                        block.write_out.weight.normal_(std=0.03)
                    reference.decoder.norm.weight.copy_(torch.linspace(0.4, 1.7, reference.config.hidden_size))
                actual.load_state_dict(reference.state_dict())
                data = batch(reference, tok, d, pairs)
                outputs, optimizers = [], []
                for m, legacy in ((reference, True), (actual, False)):
                    m.train()
                    opt = torch.optim.SGD(trainable_param_groups(m, 1e-5, 1e-3))
                    opt.zero_grad(set_to_none=True)
                    with set_checkpoint_early_stop(early_stop):
                        z = m._forward_batch(data) if legacy else decision_logits(m, data, groups)
                        loss, _, _ = step_loss(cfg, pairs, data, z, d)
                    loss.backward()
                    outputs.append(z.detach())
                    optimizers.append(opt)
                    assert all(block.write_out.weight.grad.abs().sum() > 0 for block in m.blocks)
                    assert m.get_input_embeddings().rows.grad.abs().sum() > 0
                torch.testing.assert_close(outputs[0], outputs[1], atol=3e-6, rtol=3e-5)
                for (name, p), (other, q) in zip(reference.named_parameters(), actual.named_parameters()):
                    assert name == other and (p.grad is None) == (q.grad is None), name
                    if p.grad is not None:
                        assert torch.isfinite(q.grad).all(), name
                        torch.testing.assert_close(p.grad, q.grad, atol=3e-5, rtol=3e-4, msg=name)
                for opt in optimizers:
                    opt.step()
                for (name, p), (_, q) in zip(reference.named_parameters(), actual.named_parameters()):
                    torch.testing.assert_close(p, q, atol=3e-6, rtol=3e-5, msg=name)
                assert all(not layer._forward_hooks for layer in actual.decoder.layers)


def test_failed_feedback_forward_leaves_pending_backward_and_foreign_hooks_intact():
    m, tok, d, _ = tiny_model('minimal', True, grad_ckpt=True, trainable='full')
    data = batch(m, tok, d)
    calls = []
    handle = m.decoder.layers[0].register_forward_hook(lambda *_: calls.append(1))
    try:
        want = m.eval().forward_batch(data).detach()
        pending = -m.train().forward_batch(data).log_softmax(-1)[:, d[0]].mean()
        with patch.object(m.blocks[1], 'forward', side_effect=RuntimeError('injected feedback failure')):
            try:
                m.forward_batch(data)
            except RuntimeError as exc:
                assert str(exc) == 'injected feedback failure'
            else:
                raise AssertionError('failure did not reach feedback block')
        pending.backward()
        assert m.blocks[0].write_out.weight.grad.abs().sum() > 0
        assert list(m.decoder.layers[0]._forward_hooks) == [handle.id]
        assert all(not layer._forward_hooks for layer in m.decoder.layers[1:])
        torch.testing.assert_close(m.eval().forward_batch(data), want, atol=0, rtol=0)
    finally:
        handle.remove()


def test_feedback_backward_ignores_another_threads_no_grad_forward_taps():
    for family in ('qwen3', 'qwen35'):
        m, tok, d, _ = tiny_model('minimal', True, grad_ckpt=True, family=family, trainable='full')
        with torch.no_grad():
            for block in m.blocks:
                block.write_out.weight.normal_(std=0.03)
        m.train()
        a = batch(m, tok, d)
        changed = [replace(ex, query='green short', option_names=list(reversed(ex.option_names)))
                   for ex in examples()]
        b = batch(m, tok, d, changed)
        with torch.no_grad():
            want = m.forward_batch(b)
        pending = -m.forward_batch(a).log_softmax(-1)[:, d[0]].mean()
        reached, release = threading.Event(), threading.Event()
        results, errors = [], []

        def pause(_module, _args):
            reached.set()
            assert release.wait(10), 'concurrent forward was not released'

        def forward_b():
            try:
                with torch.no_grad():
                    results.append(m.forward_batch(b))
            except Exception as exc:
                errors.append(exc)

        handle = m.decoder.norm.register_forward_pre_hook(pause)
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
        torch.testing.assert_close(results[0], want, atol=0, rtol=0)
        assert m.blocks[0].write_out.weight.grad.abs().sum() > 0
        assert all(not layer._forward_hooks for layer in m.decoder.layers)


def test_feedback_backbone_copy_and_pickle_preserve_independent_checkpoint_functions():
    m, tok, d, _ = tiny_model('minimal', True, grad_ckpt=True, trainable='full')
    data = batch(m, tok, d)
    # A checkpoint callback that captures the owning DecisionModel also captures
    # its RLock, breaking serialization of otherwise ordinary backbone layers.
    clones = [copy.deepcopy(m.base), copy.deepcopy(m.base)]
    clones[1].model.layers[-1] = pickle.loads(pickle.dumps(m.decoder.layers[-1]))
    for clone in clones:
        clone.eval()
        with torch.no_grad():
            expected = m.base(input_ids=data['input_ids'], attention_mask=data['attention_mask']).logits
            actual = clone(input_ids=data['input_ids'], attention_mask=data['attention_mask']).logits
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == '__main__':
    run(globals())
