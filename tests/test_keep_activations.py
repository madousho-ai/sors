"""Leading text layers can keep their activations instead of replaying in backward."""

import torch

from _runner import run
from test_decision import batch, tiny_backbone, tiny_model, tokenizer


def _replays(m, layers, run_backward):
    events, handles = [], []
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(lambda _m, _args, i=i: events.append(i)))
    try:
        run_backward(events)
    finally:
        for handle in handles:
            handle.remove()
    return events


def _minimal_grads(keep, family, feedback):
    from sors.core.model import keep_layer_activations

    m, tok, d, _ = tiny_model('minimal', feedback, grad_ckpt=True, family=family, trainable='full')
    if feedback:
        with torch.no_grad():
            for block in m.blocks:
                block.write_out.weight.normal_(std=0.03)
    keep_layer_activations(m, keep)
    m.train()
    data = batch(m, tok, d)

    def step(events):
        logits = m.forward_batch(data)
        events.clear()
        (-logits.log_softmax(-1)[:, d[0]].mean()).backward()

    replayed = _replays(m, m.decoder.layers, step)
    return replayed, {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}


def test_kept_minimal_layers_skip_their_backward_replay_and_keep_gradients():
    """抓住没关掉前几层重算, 或关掉时把反馈层也一起关、让候选状态断开的实现。
    tiny Qwen3 四层 (反馈层 2, 3); tiny Qwen3.5 两层, 只读路径上留第 0 层。"""
    for family, feedback, depth, keep in (('qwen3', True, 4, 2), ('qwen3', False, 4, 3), ('qwen35', False, 2, 1)):
        base_replay, base = _minimal_grads(0, family, feedback)
        kept_replay, kept = _minimal_grads(keep, family, feedback)
        assert base_replay == list(reversed(range(depth))), base_replay
        assert kept_replay == list(reversed(range(keep, depth))), (family, kept_replay)
        assert base.keys() == kept.keys() and len(base) > 0
        for name in base:
            torch.testing.assert_close(kept[name], base[name], rtol=1e-6, atol=1e-7, msg=name)


def test_plain_language_model_keeps_leading_layers_with_the_same_gradients():
    from sors.core.model import keep_layer_activations, last_logits, prepare_model, text_layers
    from sors.core.batch import collate
    from test_decision import examples

    grads, replays = [], []
    for keep in (0, 1):
        tok, d, ids = tokenizer()
        m = prepare_model(tiny_backbone(tok), ids, 4, 8, 0.0, trainable='full', grad_ckpt=True)
        keep_layer_activations(m, keep)
        m.train()
        data = collate(examples(), tok, d, 4)

        def step(events):
            logits = last_logits(m, data['input_ids'], data['attention_mask'])
            events.clear()
            logits[:, d[0]].sum().backward()

        replays.append(_replays(m, text_layers(m), step))
        grads.append({n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None})
    assert replays == [[3, 2, 1, 0], [3, 2, 1]], replays
    for name in grads[0]:
        torch.testing.assert_close(grads[1][name], grads[0][name], rtol=1e-6, atol=1e-7, msg=name)


def test_keeping_layers_is_refused_where_it_would_change_the_computation():
    """反馈层的重算函数承载候选状态, structural 走选项分块重算; 两者都不允许; 没开 grad_ckpt 也拒绝。"""
    from sors.core.model import keep_layer_activations

    cases = [(tiny_model('minimal', True, grad_ckpt=True, trainable='full')[0], 3),
             (tiny_model('structural', False, grad_ckpt=True, trainable='full')[0], 1),
             (tiny_model('minimal', False, grad_ckpt=False, trainable='full')[0], 1),
             (tiny_model('minimal', False, grad_ckpt=True, trainable='full')[0], -1),
             (tiny_model('minimal', False, grad_ckpt=True, trainable='full')[0], 5)]
    for m, keep in cases:
        try:
            keep_layer_activations(m, keep)
        except ValueError:
            continue
        raise AssertionError(f"keep={keep} accepted for {m.decision_config}")


if __name__ == "__main__":
    run(globals())
