"""Candidate-factorized decisions: joint pretrained encoding and optional sets."""

from dataclasses import replace

import torch

from _runner import run
from test_decision import examples, tiny_backbone, tokenizer
from decidophobia.core.batch import collate
from decidophobia.core.decision import DecisionConfig
from decidophobia.core.menu import reorder_menu
from decidophobia.core.model import decision_logits, prepare_model


def candidate_model(blocks=2, family="qwen3", trainable="decision-only", checkpointed=False):
    try:
        cfg = DecisionConfig(kind="candidate", blocks=blocks, dim=16, heads=4, option_batch_size=2)
    except ValueError as exc:
        raise AssertionError(f"candidate architecture is unavailable: {exc}") from exc
    tok, d, ids = tokenizer()
    m = prepare_model(tiny_backbone(tok, family), ids, 4, 8, 0., trainable=trainable,
                      grad_ckpt=checkpointed, decision=cfg)
    return m, tok, d, ids


def candidate_batch(tok, d, exs=None, **kw):
    return collate(exs or examples(), tok, d, 4, architecture="candidate", **kw)


def joint_text(ex, name):
    context = f"{ex.context_label}: {ex.query}" if ex.context_label else ex.query
    return f"{context}\n\nQuestion: {ex.question}\n\nOption: {name}\n\nAnswer:"


def test_each_candidate_contains_the_complete_context_question_and_only_its_option():
    _, tok, d, _ = candidate_model(0)
    ex = replace(examples()[0], codes=[200, 78, 9])
    b = candidate_batch(tok, d, [ex])
    for i, option in enumerate(ex.option_names):
        ids = b["option_input_ids"][0, i][b["option_attention_mask"][0, i].bool()].tolist()
        want = tok.encode(joint_text(ex, option), add_special_tokens=False, split_special_tokens=True)
        assert ids == want
        assert not set(ids) & set(d), "output codes leaked into semantic encoding"
    assert b["slot_ids"][0].tolist() == [d[200], d[78], d[9], -1]


def test_zero_set_blocks_is_exactly_pretrained_joint_features_and_shared_rms_scalar_score():
    for family in ("qwen3", "qwen35"):
        m, tok, d, _ = candidate_model(0, family)
        m.eval()
        ex = examples()[0]
        b = candidate_batch(tok, d, [ex])
        got = m.forward_batch(b)[0, b["slot_ids"][0, :3]]
        want = []
        for option in ex.option_names:
            ids = torch.tensor([tok.encode(joint_text(ex, option), add_special_tokens=False)])
            h = m.decoder(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state[:, -1].float()
            norm = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + 1e-6) * m.option_norm.weight
            want.append((norm * m.scorer.weight[0]).sum())
        torch.testing.assert_close(got, torch.stack(want), atol=2e-6, rtol=2e-5)
        assert len(m.blocks) == 0


def test_candidate_does_not_truncate_context_to_fit_a_branch():
    _, tok, d, _ = candidate_model()
    ex = replace(examples()[0], query="long context " * 30)
    for context_marker in (False, True):
        try:
            candidate_batch(tok, d, [ex], max_length=20, context_marker=context_marker)
        except ValueError as exc:
            assert "candidate" in str(exc) and "max_length" in str(exc)
        else:
            raise AssertionError("a candidate silently lost part of C,Q,option")


def test_zero_one_two_set_blocks_are_equivariant_on_both_backbones():
    for family in ("qwen3", "qwen35"):
        for count in (0, 1, 2):
            m, tok, d, _ = candidate_model(count, family)
            m.eval()
            ex = examples()[0]
            other = reorder_menu(ex, [2, 0, 1], [155, 0, 200])
            a = candidate_batch(tok, d, [ex])
            b = candidate_batch(tok, d, [other])
            p = m.forward_batch(a)[0, a["slot_ids"][0, :3]].softmax(-1)
            q = m.forward_batch(b)[0, b["slot_ids"][0, :3]].softmax(-1)
            torch.testing.assert_close(q, p[[2, 0, 1]], atol=2e-6, rtol=2e-5)
            assert len(m.blocks) == count
            assert all(not layer._forward_hooks for layer in m.decoder.layers)


def test_set_interaction_changes_evidence_while_zero_blocks_score_options_independently():
    ex = examples()[0]
    changed = replace(ex, option_names=["red", "green", "long blue"])
    for count in (0, 1, 2):
        m, tok, d, _ = candidate_model(count)
        m.eval()
        a = m.forward_batch(candidate_batch(tok, d, [ex]))[0, d[0]]
        b = m.forward_batch(candidate_batch(tok, d, [changed]))[0, d[0]]
        if count == 0:
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
        else:
            assert abs((a - b).item()) > 1e-5, "candidate interaction is disconnected"


def test_frozen_candidate_still_reads_context_and_trains_the_scalar_head():
    m, tok, d, _ = candidate_model(0)
    a = examples()[0]
    b = replace(a, query="blue context")
    pa = m.forward_batch(candidate_batch(tok, d, [a])).softmax(-1)
    pb = m.forward_batch(candidate_batch(tok, d, [b])).softmax(-1)
    assert not torch.allclose(pa, pb, atol=1e-8, rtol=1e-6)
    frozen = {n: p.detach().clone() for n, p in m.base.named_parameters() if not n.endswith(".rows")}
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.01)
    loss = -pa[0, d[0]].log()
    loss.backward()
    assert m.scorer.weight.grad.abs().sum() > 0
    opt.step()
    for n, p in m.base.named_parameters():
        if n in frozen:
            assert p.grad is None and torch.equal(p, frozen[n]), n


def test_candidate_padding_grouping_and_chunking_preserve_scores():
    m, tok, d, _ = candidate_model(2)
    m.eval()
    data = candidate_batch(tok, d)
    together = m.forward_batch(data)
    chunked = m.forward_batch(data, option_batch_size=1)
    grouped = decision_logits(m, data, groups=2)
    torch.testing.assert_close(together, chunked, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(together, grouped, atol=2e-6, rtol=2e-5)
    changed = {k: v.clone() for k, v in data.items()}
    changed["option_input_ids"][changed["slot_ids"] < 0] = 10
    torch.testing.assert_close(m.forward_batch(changed), together, atol=0, rtol=0)


def test_candidate_rejects_cross_layer_feedback_and_tap_indices():
    candidate_model(0)
    for extra in ({"feedback": True}, {"layers": (0, 1)}):
        try:
            DecisionConfig(kind="candidate", **extra).resolve(4)
        except ValueError as exc:
            assert "candidate" in str(exc)
        else:
            raise AssertionError("candidate accepted a cross-layer configuration")
    assert DecisionConfig(kind="candidate", blocks=2).resolve(1).layers == ()


def test_candidate_forwards_exactly_the_joint_branches_without_an_extra_prefix_pass():
    m, tok, d, _ = candidate_model(2)
    m.set_candidate_prefix_cache("off")  # Explicit complete-forward reference.
    calls = []
    handle = m.decoder.register_forward_pre_hook(
        lambda module, args, kwargs: calls.append(kwargs["input_ids"].shape[0]), with_kwargs=True)
    try:
        m.forward_batch(candidate_batch(tok, d))
    finally:
        handle.remove()
    assert calls == [2, 2, 1], calls


def test_candidate_checkpointing_preserves_full_backbone_gradients_in_both_dtypes():
    for family in ("qwen3", "qwen35"):
        for dtype in (torch.float32, torch.bfloat16):
            tok, d, ids = tokenizer()
            cfg = DecisionConfig(kind="candidate", blocks=2, dim=16, heads=4, option_batch_size=2)
            m = prepare_model(tiny_backbone(tok, family).to(dtype), ids, 4, 8, 0., trainable="full", decision=cfg)
            gradients = []
            for checkpointed in (False, True):
                m.checkpoint_forward = checkpointed
                m.train()
                m.zero_grad(set_to_none=True)
                loss = -m.forward_batch(candidate_batch(tok, d)).log_softmax(-1)[:, d[0]].mean()
                loss.backward()
                grads = {n: p.grad.detach().float().clone() for n, p in m.named_parameters() if p.grad is not None}
                assert all(torch.isfinite(g).all() for g in grads.values())
                assert grads["scorer.weight"].abs().sum() > 0
                assert grads["blocks.0.attention.in_proj_weight"].abs().sum() > 0
                assert any(n.startswith("base.") and g.abs().sum() > 0 for n, g in grads.items())
                gradients.append(grads)
            assert gradients[0].keys() == gradients[1].keys()
            for name, grad in gradients[0].items():
                torch.testing.assert_close(grad, gradients[1][name], atol=3e-5, rtol=3e-4, msg=name)


def test_candidate_lora_receives_gradients_through_joint_encoding():
    m, tok, d, _ = candidate_model(0, trainable="attn", checkpointed=True)
    (-m.forward_batch(candidate_batch(tok, d)).log_softmax(-1)[:, d[0]].mean()).backward()
    assert any("lora_B" in name and p.grad is not None and p.grad.abs().sum() > 0 for name, p in m.named_parameters())
    assert m.scorer.weight.grad.abs().sum() > 0


def test_set_blocks_can_outnumber_backbone_layers():
    from transformers import Qwen3ForCausalLM
    tok, d, ids = tokenizer()
    config = tiny_backbone(tok).config
    config.num_hidden_layers = 1
    config.layer_types = ["full_attention"]
    m = prepare_model(Qwen3ForCausalLM(config), ids, 4, 8, 0., trainable="decision-only",
                      decision=DecisionConfig(kind="candidate", blocks=2, dim=16, heads=4))
    p = m.forward_batch(candidate_batch(tok, d)).softmax(-1)
    assert torch.isfinite(p).all()
    torch.testing.assert_close(p.sum(-1), torch.ones(2))


def test_candidate_all_256_choices_share_the_same_scorer():
    m, tok, d, _ = candidate_model(2)
    ex = replace(examples()[0], options=list(range(256)), label=0, gold_idx=0,
                 option_names=["red"] * 256, codes=list(reversed(range(256))))
    data = collate([ex], tok, d, 256, architecture="candidate")
    m.eval()
    p = m.forward_batch(data).softmax(-1)[0, d]
    torch.testing.assert_close(p, torch.full((256,), 1 / 256), atol=1e-7, rtol=1e-5)


def test_candidate_real_tokenizer_preserves_joint_text_and_protects_reserved_tokens():
    from test_batch_loss import _all_tokens
    tok, d, t, c = _all_tokens()
    ex = replace(examples()[0], query="上下文 <|D7|> 保留", option_names=["你好 🔴", "literal <|D9|>", "最后一个!"], codes=[222, 100, 8])
    data = candidate_batch(tok, d, [ex], context_marker=True, type_marker=True)
    for i, name in enumerate(ex.option_names):
        ids = data["option_input_ids"][0, i][data["option_attention_mask"][0, i].bool()].tolist()
        want = (f"<|context_start|>Customer message: {ex.query}<|context_end|>\n\n"
                f"Question (<|choice|>): {ex.question}\n\nOption: {name}\n\nAnswer:")
        assert tok.decode(ids) == want
        assert not set(ids) & set(d)
        assert ids.count(t[0]) == 1 and ids.count(c[0]) == 1 and ids.count(c[1]) == 1


if __name__ == "__main__":
    run(globals())
