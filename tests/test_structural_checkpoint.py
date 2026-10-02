"""Read-only structural checkpoints bound saved activations and preserve training."""
from dataclasses import replace
import random
import threading
from unittest.mock import patch

import torch
from torch.utils.checkpoint import set_checkpoint_early_stop

from _runner import run
from test_decision import batch, examples, tiny_backbone, tiny_model, tokenizer
from sors.core.batch import length_groups
from sors.core.decision import DecisionConfig
from sors.core.menu import with_partners
from sors.core.model import prepare_model, select_batch, trainable_param_groups
from sors.training.loop import TrainConfig, step_loss


def long_examples():
    return [replace(ex, query=ex.query + ' red context' * (9 + i)) for i, ex in enumerate(examples())]


def grouped(m, data, count, chunk, legacy):
    rows = length_groups(data['attention_mask'], count)
    forward = m._forward_batch if legacy else m.forward_batch
    parts = [forward(select_batch(data, group), option_batch_size=chunk) for group in rows]
    return torch.cat(parts)[torch.argsort(torch.cat(rows))]


def test_option_chunks_save_inputs_instead_of_all_decoder_activations():
    for family in ('qwen3', 'qwen35'):
        m, tok, d, _ = tiny_model('structural', False, grad_ckpt=True, family=family, trainable='full')
        data = batch(m, tok, d)
        saved = []

        def pack(tensor):
            saved.append((tensor.is_floating_point(), tensor.numel()))
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            encoded = m.train()._encode_options(data, data['slot_ids'] >= 0, torch.float32, 1)
        assert not any(floating and size for floating, size in saved), (family, sum(n for f, n in saved if f))
        encoded.square().sum().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.decoder.parameters())


def test_structural_common_stream_recomputes_layers_in_reverse_and_blocks_separately():
    for family in ('qwen3', 'qwen35'):
        m, tok, d, _ = tiny_model('structural', False, grad_ckpt=True, family=family, trainable='full')
        data = batch(m, tok, d, long_examples())
        width = data['input_ids'].shape[1]
        assert width > data['option_input_ids'].shape[-1]
        events, handles = [], []
        for i, layer in enumerate(m.decoder.layers):
            def record(_module, args, index=i):
                if args[0].shape[1] == width:
                    events.append(('text', index))
            handles.append(layer.register_forward_pre_hook(record))
        for i, block in enumerate(m.blocks):
            handles.append(block.register_forward_pre_hook(lambda _m, _a, i=i: events.append(('decision', i))))
        try:
            z = m.train().forward_batch(data)
            assert events == [('text', i) for i in range(len(m.decoder.layers))] + [('decision', 0), ('decision', 1)], events
            events.clear()
            (-z.log_softmax(-1)[:, d[0]].mean()).backward()
            assert [i for kind, i in events if kind == 'text'] == list(reversed(range(len(m.decoder.layers)))), events
            assert [i for kind, i in events if kind == 'decision'] == [1, 0], events
        finally:
            for handle in handles:
                handle.remove()


def test_structural_matches_legacy_logits_loss_all_gradients_and_sgd_updates():
    pairs = with_partners(long_examples(), random.Random(0))
    cfg = TrainConfig(loss='menu', consistency=1.0)
    for family in ('qwen3', 'qwen35'):
        for trainable in ('full', 'decision-only', 'attn'):
            for checkpointed, groups, chunk in ((False, 1, None), (True, 1, None), (True, 2, 1)):
                reference, tok, d, _ = tiny_model('structural', False, family=family, trainable=trainable)
                actual, _, _, _ = tiny_model('structural', False, grad_ckpt=checkpointed,
                                            family=family, trainable=trainable)
                with torch.no_grad():
                    reference.decoder.norm.weight.copy_(torch.linspace(0.4, 1.7, reference.config.hidden_size))
                actual.load_state_dict(reference.state_dict())
                data = batch(reference, tok, d, pairs, type_marker=True, context_marker=True)
                values, optimizers = [], []
                for m, legacy in ((reference, True), (actual, False)):
                    m.train()
                    opt = torch.optim.SGD(trainable_param_groups(m, 1e-5, 1e-3))
                    opt.zero_grad(set_to_none=True)
                    z = grouped(m, data, groups, chunk, legacy)
                    loss, _, _ = step_loss(cfg, pairs, data, z, d)
                    loss.backward()
                    values.append((z.detach(), loss.detach()))
                    optimizers.append(opt)
                    assert m.blocks[0].read.in_proj_weight.grad.abs().sum() > 0
                    assert m.get_input_embeddings().rows.grad.abs().sum() > 0
                torch.testing.assert_close(values[0], values[1], atol=3e-6, rtol=3e-5)
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


def test_structural_checkpoints_preserve_lora_dropout_rng_and_gradients():
    tok, d, ids = tokenizer()
    config = DecisionConfig(kind='structural', dim=16, heads=4, blocks=2, option_batch_size=2)
    models = [prepare_model(tiny_backbone(tok), ids, 4, 8, 0.2, trainable='attn', grad_ckpt=enabled,
                            decision=config) for enabled in (False, True)]
    with torch.no_grad():
        for name, p in models[0].named_parameters():
            if 'lora_B' in name:
                p.normal_(std=0.03)
    models[1].load_state_dict(models[0].state_dict())
    data = batch(models[0], tok, d, long_examples(), type_marker=True)
    outputs, rng_states = [], []
    for i, m in enumerate(models):
        m.train()
        torch.manual_seed(912)
        z = m._forward_batch(data) if i == 0 else m.forward_batch(data)
        (-z.log_softmax(-1)[:, d[0]].mean()).backward()
        outputs.append(z.detach())
        rng_states.append(torch.get_rng_state())
    torch.testing.assert_close(outputs[0], outputs[1], atol=3e-6, rtol=3e-5)
    assert torch.equal(rng_states[0], rng_states[1])
    for (name, p), (_, q) in zip(models[0].named_parameters(), models[1].named_parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=3e-5, rtol=3e-4, msg=name)


def test_structural_matches_legacy_adamw_updates_in_double_precision():
    for family in ('qwen3', 'qwen35'):
        reference, tok, d, _ = tiny_model('structural', False, family=family, trainable='full')
        actual, _, _, _ = tiny_model('structural', False, grad_ckpt=True, family=family, trainable='full')
        actual.load_state_dict(reference.state_dict())
        models = [m.double().train() for m in (reference, actual)]
        optimizers = [torch.optim.AdamW(trainable_param_groups(m, 1e-5, 1e-3), weight_decay=0) for m in models]
        data = batch(reference, tok, d, long_examples(), type_marker=True, context_marker=True)
        for _ in range(2):
            for index, (m, optimizer) in enumerate(zip(models, optimizers)):
                optimizer.zero_grad(set_to_none=True)
                z = m._forward_batch(data) if index == 0 else m.forward_batch(data)
                (-z.log_softmax(-1)[:, d[0]].mean()).backward()
                optimizer.step()
            for (name, p), (_, q) in zip(reference.named_parameters(), actual.named_parameters()):
                torch.testing.assert_close(p, q, atol=1e-8, rtol=1e-6, msg=name)


def test_structural_common_capture_cleans_up_after_failure():
    m, tok, d, _ = tiny_model('structural', False, grad_ckpt=True, trainable='full')
    data = batch(m, tok, d, long_examples())
    width = data['input_ids'].shape[1]
    assert width > data['option_input_ids'].shape[-1]
    foreign = m.decoder.layers[0].register_forward_hook(lambda *_: None)
    forward = m.decoder.layers[-1].forward

    def fail_common(h, *args, **kwargs):
        if h.shape[1] == width:
            raise RuntimeError('injected common-stream failure')
        return forward(h, *args, **kwargs)

    try:
        expected = m.eval().forward_batch(data).detach()
        with patch.object(m.decoder.layers[-1], 'forward', side_effect=fail_common):
            try:
                m.train().forward_batch(data)
            except RuntimeError as exc:
                assert str(exc) == 'injected common-stream failure'
            else:
                raise AssertionError('expected an injected encoder failure')
        assert list(m.decoder.layers[0]._forward_hooks) == [foreign.id]
        assert all(not layer._forward_hooks for layer in m.decoder.layers[1:])
        torch.testing.assert_close(m.eval().forward_batch(data), expected, atol=0, rtol=0)
    finally:
        foreign.remove()


def test_structural_capture_ignores_other_thread_option_and_common_replays():
    for family in ('qwen3', 'qwen35'):
        m, tok, d, _ = tiny_model('structural', False, grad_ckpt=True, family=family, trainable='full')
        a = batch(m, tok, d, long_examples())
        changed = [replace(ex, option_names=list(reversed(ex.option_names))) for ex in long_examples()]
        b = batch(m, tok, d, changed)
        with torch.no_grad():
            expected = m.eval().forward_batch(b)
        m.train()
        with set_checkpoint_early_stop(False):
            pending = -m.forward_batch(a).log_softmax(-1)[:, d[0]].mean()
        reached, release = threading.Event(), threading.Event()
        result, errors = [], []

        def pause_common(_module, args):
            if args[0].shape[1] == b['input_ids'].shape[1]:
                reached.set()
                assert release.wait(10), 'forward thread was not released'

        def forward_b():
            try:
                result.append(m.forward_batch(b).detach())
            except Exception as exc:
                errors.append(exc)

        handle = m.decoder.norm.register_forward_pre_hook(pause_common)
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


if __name__ == '__main__':
    run(globals())
