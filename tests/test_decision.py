"""Decision layers: real attention, permutation symmetry and independent feedback."""

import importlib
import importlib.util

import torch

from _runner import run


def decision_module():
    assert importlib.util.find_spec("decidophobia.core.decision") is not None, "decision layers are missing"
    return importlib.import_module("decidophobia.core.decision")


def block(feedback=False):
    torch.manual_seed(31)
    return decision_module().DecisionBlock(16, 32, 4, feedback=feedback).eval()


def inputs():
    torch.manual_seed(9)
    return (torch.randn(2, 3, 16), torch.randn(2, 5, 32),
            torch.tensor([[True, True, True], [True, True, False]]),
            torch.tensor([[True] * 5, [False, False, True, True, True]]))


def test_decision_block_is_equivariant_with_learned_feedback():
    m = block(True)
    with torch.no_grad():
        m.write_out.weight.normal_(std=0.1)
    u, h, om, hm = inputs()
    p = torch.tensor([2, 0, 1])
    a, ah = m(u, h, om, hm)
    b, bh = m(u[:, p], h, om[:, p], hm)
    torch.testing.assert_close(b, a[:, p], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(bh, ah, atol=2e-6, rtol=2e-5)
    assert not torch.allclose(ah, h), "feedback never reached the text stream"


def test_feedback_starts_as_identity_but_has_a_learning_signal():
    m = block(True)
    u, h, om, hm = inputs()
    _, got = m(u, h, om, hm)
    torch.testing.assert_close(got, h, rtol=0, atol=0)
    got.square().sum().backward()
    assert m.write_out.weight.grad.abs().sum() > 0


def test_feedback_off_leaves_text_unchanged_after_training_the_decision_block():
    m = block(False)
    u, h, om, hm = inputs()
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    out, _ = m(u, h, om, hm)
    out.square().sum().backward()
    opt.step()
    out, got = m(u, h, om, hm)
    torch.testing.assert_close(got, h, rtol=0, atol=0)
    assert not torch.allclose(out[om], u[om])


def test_padding_cannot_contribute_evidence_or_candidates():
    m = block(True)
    with torch.no_grad():
        m.write_out.weight.normal_(std=0.1)
    u, h, om, hm = inputs()
    a, ah = m(u, h, om, hm)
    changed_u, changed_h = u.clone(), h.clone()
    changed_u[~om] = 500
    changed_h[~hm] = -500
    b, bh = m(changed_u, changed_h, om, hm)
    torch.testing.assert_close(a[om], b[om])
    torch.testing.assert_close(ah[hm], bh[hm])
    assert torch.count_nonzero(b[~om]) == 0


def test_decision_reads_context_and_other_options_with_finite_gradients():
    m = block()
    u, h, om, hm = inputs()
    u.requires_grad_()
    h.requires_grad_()
    out, _ = m(u, h, om, hm)
    out[0, 0].square().sum().backward()
    assert h.grad[0].abs().sum() > 0, "context connection is dead"
    assert u.grad[0, 1:].abs().sum() > 0, "candidate comparison is dead"
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from decidophobia.core.tokens import install_context_tokens, install_d_tokens, install_type_tokens

    words = ["[PAD]", "[UNK]", "red", "blue", "green", "short", "long", "context", "Question",
             "Answer", "Options", "Option", "Which", "color", "Customer", "message", ":", ".", "?"]
    backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]")
    d = install_d_tokens(tok)
    train_ids = d + install_type_tokens(tok) + install_context_tokens(tok)
    return tok, d, train_ids


def examples():
    from decidophobia.core.menu import MenuExample
    return [MenuExample("red context", [10, 20, 30], 0, 10, ["red", "blue", "green"], question="Which color?"),
            MenuExample("long blue context context", [10, 20], 1, 20, ["red", "blue"], question="Which color?",
                        target=[0.2, 0.8])]


def tiny_model(kind="structural", feedback=False, grad_ckpt=False, family="qwen3", trainable="decision-only"):
    from decidophobia.core.model import prepare_model

    module = decision_module()
    assert hasattr(module, "DecisionConfig"), "decision model configuration is missing"
    tok, d, ids = tokenizer()
    lm = tiny_backbone(tok, family)
    config = module.DecisionConfig(kind=kind, dim=16, heads=4, blocks=2, feedback=feedback, option_batch_size=2)
    return prepare_model(lm, ids, 4, 8, 0.0, trainable=trainable, grad_ckpt=grad_ckpt,
                         decision=config), tok, d, ids


def tiny_backbone(tok, family="qwen3"):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(43)
    if family == "qwen3":
        lm = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tok) + 4, hidden_size=32, intermediate_size=64,
                                        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                                        head_dim=8, use_cache=False, attention_dropout=0.0))
    else:
        from test_qwen35 import tiny_qwen35
        lm = tiny_qwen35()
    return lm


def batch(m, tok, d, exs=None, **kwargs):
    from decidophobia.core.batch import collate
    return collate(exs or examples(), tok, d, 4, architecture=m.decision_config.kind, **kwargs)


def test_minimal_batch_preserves_original_prompt_and_tracks_description_ends():
    from decidophobia.core.batch import collate
    m, tok, d, _ = tiny_model("minimal")
    for layout in ("context-first", "menu-first"):
        b = batch(m, tok, d, layout=layout)
        old = collate(examples(), tok, d, 4, layout=layout)
        torch.testing.assert_close(b["input_ids"], old["input_ids"])
        for row, ex in enumerate(examples()):
            for j, text in enumerate(ex.option_names):
                assert b["input_ids"][row, b["option_positions"][row, j]].item() == tok.convert_tokens_to_ids(text)
    assert b["option_positions"][1, 2:].tolist() == [-1, -1]


def test_minimal_refuses_truncation_that_removes_a_candidate():
    m, tok, d, _ = tiny_model("minimal")
    try:
        batch(m, tok, d, max_length=4)
    except ValueError as exc:
        assert "option" in str(exc)
        return
    raise AssertionError("a truncated option silently received a probability")


def test_structural_inputs_isolate_options_from_order_and_code():
    from decidophobia.core.menu import reorder_menu
    m, tok, d, _ = tiny_model()
    ex = examples()[0]
    other = reorder_menu(ex, [2, 0, 1], [200, 90, 155])
    a, b = batch(m, tok, d, [ex]), batch(m, tok, d, [other])
    torch.testing.assert_close(a["input_ids"], b["input_ids"])
    torch.testing.assert_close(b["option_input_ids"][:, :3], a["option_input_ids"][:, [2, 0, 1]])
    assert not torch.isin(a["input_ids"], torch.tensor(d)).any()
    assert not torch.isin(a["option_input_ids"], torch.tensor(d)).any()
    assert a["target"][0].tolist() == [1, 0, 0, 0]


def test_both_architectures_produce_menu_probabilities_and_train_new_layers_on_both_backbones():
    for family in ("qwen3", "qwen35"):
        for kind in ("minimal", "structural"):
            for feedback in (False, True):
                m, tok, d, _ = tiny_model(kind, feedback, family=family)
                m.train()
                b = batch(m, tok, d)
                z = m.forward_batch(b)
                assert z.shape == (2, m.config.vocab_size)
                logits = z.gather(1, b["slot_ids"].clamp_min(0))
                p = z.softmax(-1)
                torch.testing.assert_close(p.sum(1), torch.ones(2))
                loss = -torch.log_softmax(logits[:, :2], -1)[:, 0].mean()
                loss.backward()
                assert torch.isfinite(loss)
                assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
                assert m.blocks[0].read.in_proj_weight.grad.abs().sum() > 0
                if feedback:
                    assert all(block.write_out.weight.grad.abs().sum() > 0 for block in m.blocks)
                assert not any(p.requires_grad for n, p in m.base.named_parameters() if not n.endswith(".rows"))


def test_structural_model_is_equivariant_after_feedback_learns():
    from decidophobia.core.menu import reorder_menu
    for family in ("qwen3", "qwen35"):
        m, tok, d, _ = tiny_model(feedback=True, family=family)
        m.eval()
        with torch.no_grad():
            for layer in m.blocks:
                layer.write_out.weight.normal_(std=0.03)
        ex = examples()[0]
        changed = reorder_menu(ex, [2, 0, 1], [200, 90, 155])
        a, b = batch(m, tok, d, [ex]), batch(m, tok, d, [changed])
        za, zb = m.forward_batch(a), m.forward_batch(b)
        pa = za[0, a["slot_ids"][0, :3]].softmax(-1)
        pb = zb[0, b["slot_ids"][0, :3]].softmax(-1)
        torch.testing.assert_close(pb, pa[[2, 0, 1]], atol=2e-6, rtol=2e-5)


def test_checkpointed_decision_forward_matches_gradients_and_cleans_up_hooks():
    for family, kind in ((family, kind) for family in ("qwen3", "qwen35") for kind in ("minimal", "structural")):
        a, tok, d, _ = tiny_model(kind, True, family=family, trainable="full")
        with torch.no_grad():
            for layer in a.blocks:
                layer.write_out.weight.normal_(std=0.03)
        b, _, _, _ = tiny_model(kind, True, grad_ckpt=True, family=family, trainable="full")
        b.load_state_dict(a.state_dict())
        a.train()
        b.train()
        data = batch(a, tok, d)
        for m in (a, b):
            for _ in range(2):
                m.zero_grad(set_to_none=True)
                z = m.forward_batch(data)
                (-z.log_softmax(-1)[:, d[0]].mean()).backward()
            assert all(not layer._forward_hooks for layer in m.decoder.layers)
        for (n, p), (other_n, q) in zip(a.named_parameters(), b.named_parameters()):
            assert n == other_n
            assert (p.grad is None) == (q.grad is None), n
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, atol=3e-5, rtol=3e-4, msg=n)


def test_real_tokenizer_preserves_minimal_prompts_markers_and_truncated_context():
    from dataclasses import replace
    from test_batch_loss import _all_tokens
    from decidophobia.core.batch import collate
    tok, d, _, _ = _all_tokens()
    ex = replace(examples()[0], query="a long context " * 100 + " <|D200|> 字符串",
                 option_names=["你好 🔴\nfirst", "literal <|D9|> option", "third option!"], codes=[200, 9, 78])
    for layout in ("context-first", "menu-first"):
        for cm in (False, True):
            b = collate([ex], tok, d, 4, layout, 4096, True, cm, architecture="minimal")
            old = collate([ex], tok, d, 4, layout, 4096, True, cm)
            torch.testing.assert_close(b["input_ids"], old["input_ids"])
            for j, name in enumerate(ex.option_names):
                start = b["input_ids"][0].tolist().index(d[ex.slot_codes[j]]) + 1
                end = b["option_positions"][0, j].item() + 1
                assert tok.decode(b["input_ids"][0, start:end]).startswith(". " + name)
    b = collate([ex], tok, d, 4, "context-first", 100, True, True, architecture="minimal")
    old = collate([ex], tok, d, 4, "context-first", 100, True, True)
    torch.testing.assert_close(b["input_ids"], old["input_ids"])
    assert bool((b["option_positions"][0, :3] >= 0).all())


def test_padding_grouping_and_option_chunking_preserve_both_architecture_outputs():
    from decidophobia.core.model import decision_logits
    for kind in ("minimal", "structural"):
        m, tok, d, _ = tiny_model(kind, True)
        m.eval()
        data = batch(m, tok, d)
        with torch.no_grad():
            for layer in m.blocks:
                layer.write_out.weight.normal_(std=0.02)
        together = decision_logits(m, data)
        grouped = decision_logits(m, data, 2)
        torch.testing.assert_close(together, grouped, atol=2e-6, rtol=2e-5)
        chunked = m.forward_batch(data, option_batch_size=1)
        torch.testing.assert_close(together, chunked, atol=2e-6, rtol=2e-5)


def test_feedback_changes_later_backbone_activations_and_decision_probabilities():
    m, tok, d, _ = tiny_model("structural", True)
    m.eval()
    data = batch(m, tok, d)
    states = []
    handle = m.decoder.layers[-1].register_forward_pre_hook(lambda layer, args: states.append(args[0].detach().clone()))
    before = m.forward_batch(data).detach()
    baseline_last_input = states[-1]
    states.clear()
    with torch.no_grad():
        m.blocks[0].write_out.weight.normal_(std=0.1)
    after = m.forward_batch(data).detach()
    handle.remove()
    assert not torch.allclose(states[-1], baseline_last_input), "writeback did not reach a later native decoder layer"
    assert not torch.allclose(before.softmax(-1), after.softmax(-1)), "writeback has no decision effect"


def test_bfloat16_backbones_and_float32_decision_layers_have_finite_gradients():
    from decidophobia.core.model import prepare_model
    for family in ("qwen3", "qwen35"):
        tok, d, ids = tokenizer()
        lm = tiny_backbone(tok, family).to(torch.bfloat16)
        cfg = decision_module().DecisionConfig(kind="structural", dim=16, heads=4, feedback=True)
        m = prepare_model(lm, ids, 4, 8, 0., trainable="full", grad_ckpt=True, decision=cfg)
        with torch.no_grad():
            for layer in m.blocks:
                layer.write_out.weight.normal_(std=0.02)
        b = batch(m, tok, d)
        loss = -m.forward_batch(b).log_softmax(-1)[:, d[0]].mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert m.blocks[0].read.in_proj_weight.grad.abs().sum() > 0
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_all_256_candidates_remain_addressable_with_sparse_output_codes():
    from dataclasses import replace
    from decidophobia.core.batch import collate
    for kind in ("minimal", "structural"):
        m, tok, d, _ = tiny_model(kind)
        ex = replace(examples()[0], options=list(range(256)), label=0, gold_idx=0,
                     option_names=["red" if i % 2 else "blue" for i in range(256)], codes=list(reversed(range(256))))
        b = collate([ex], tok, d, 256, max_length=4096, architecture=kind)
        p = m.eval().forward_batch(b).softmax(-1)
        assert torch.count_nonzero(p) == 256
        torch.testing.assert_close(p.sum(), torch.tensor(1.0))


def test_failed_forward_removes_only_its_own_taps_and_next_call_recovers():
    from unittest.mock import patch
    m, tok, d, _ = tiny_model("structural", True)
    data = batch(m, tok, d)
    before = m.eval().forward_batch(data).detach()
    with patch.object(m.blocks[1], "forward", side_effect=RuntimeError("injected failure")):
        try:
            m.forward_batch(data)
        except RuntimeError as exc:
            assert str(exc) == "injected failure"
        else:
            raise AssertionError("fault injection did not reach decision block")
    assert all(not layer._forward_hooks for layer in m.decoder.layers)
    torch.testing.assert_close(m.forward_batch(data), before, atol=0, rtol=0)


def test_single_channel_decision_is_rejected_before_layernorm_erases_candidate_information():
    try:
        decision_module().DecisionConfig(dim=1, heads=1)
    except ValueError as exc:
        assert "dim" in str(exc)
        return
    raise AssertionError("LayerNorm(1) makes every option identical; this configuration must be rejected")


def test_minimal_encoder_rejects_uninstalled_template_markers():
    from test_batch_loss import _tok
    from decidophobia.core.tokens import install_d_tokens
    from decidophobia.core.batch import collate
    tok = _tok()
    d = install_d_tokens(tok)
    for type_marker, context_marker in ((True, False), (False, True)):
        try:
            collate(examples(), tok, d, 4, type_marker=type_marker, context_marker=context_marker,
                    architecture="minimal")
        except ValueError as exc:
            assert "not installed" in str(exc)
        else:
            raise AssertionError("an uninstalled template marker was silently encoded")


if __name__ == "__main__":
    run(globals())
