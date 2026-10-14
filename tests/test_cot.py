"""Training-only working (cot): curric items carry program-written working; a language-model pass learns
to write it right after 'Answer:', with the decision layers off. Inference never sees it."""

import random

import pytest
import torch

from test_decision import tiny_model
from sors.core.menu import MenuExample, partner


def exs():
    return [MenuExample("red context", [10, 20], 1, 20, ["red", "blue"], question="Which color?",
                        cot="Working: short blue . red"),
            MenuExample("long blue context", [10, 20], 0, 10, ["red", "blue"], question="Which color?"),
            MenuExample("green context", [10, 20, 30], 2, 30, ["red", "blue", "green"], question="Which color?",
                        cot="Working: green")]


def test_curric_items_carry_cot_and_other_items_do_not():
    from sors.data.synth_v5 import item_example, load_synth_v5
    items = load_synth_v5()
    cur = [it for it in items if "_curric/" in it.context_id]
    assert len(cur) == 5758 and all(it.cot and it.cot.startswith("Let me work this out.") for it in cur)
    assert all(it.cot is None for it in items if "_curric/" not in it.context_id)
    ex = item_example(cur[0], random.Random(0), 0.5, fallback_rate=1.0)
    assert ex.cot == cur[0].cot and partner(ex, random.Random(1)).cot == cur[0].cot


def test_cot_sequences_continue_the_decision_prompt_and_label_only_the_working():
    from sors.core.batch import collate
    from sors.training.cot import cot_batch
    m, tok, d, _ = tiny_model(kind="minimal", feedback=True)
    b = cot_batch(exs(), tok, d, "context-first", 512, False, False, paired=False)
    assert b["cot_input_ids"].shape[0] == 2, "only the examples with working get a sequence"
    for row, ex in zip(range(2), (exs()[0], exs()[2])):
        mask = b["cot_attention_mask"][row].bool()
        ids, labels = b["cot_input_ids"][row][mask], b["cot_labels"][row][mask]
        prompt = collate([ex], tok, d, 4, architecture="minimal")["input_ids"][0].tolist()
        working = tok(" " + ex.cot, add_special_tokens=False)["input_ids"]
        tail = working + ([tok.eos_token_id] if tok.eos_token_id is not None else [])
        assert ids.tolist() == prompt + tail
        assert labels.tolist() == [-100] * len(prompt) + tail


def test_paired_batches_use_one_view_per_question():
    from sors.training.cot import cot_batch
    m, tok, d, _ = tiny_model(kind="minimal", feedback=True)
    pairs = [x for ex in exs() for x in (ex, partner(ex, random.Random(0)))]
    b = cot_batch(pairs, tok, d, "context-first", 512, False, False, paired=True)
    assert b["cot_input_ids"].shape[0] == 2


def test_no_working_in_the_batch_means_no_cot_tensors():
    from sors.training.cot import cot_batch
    m, tok, d, _ = tiny_model(kind="minimal", feedback=True)
    assert cot_batch([exs()[1]], tok, d, "context-first", 512, False, False, paired=False) == {}


def test_cot_loss_trains_the_backbone_but_never_the_decision_layers_or_vocabulary():
    from sors.training.cot import cot_batch, cot_lm_loss
    m, tok, d, _ = tiny_model(kind="minimal", feedback=True, grad_ckpt=True, trainable="full")
    b = cot_batch(exs(), tok, d, "context-first", 512, False, False, paired=False)
    m.train()
    loss = cot_lm_loss(m, b)
    assert loss.ndim == 0 and torch.isfinite(loss) and loss > 0
    loss.backward()
    named = dict(m.named_parameters())
    body = [p for n, p in named.items() if n.startswith("base.model.") and p.requires_grad]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in body), "the backbone got no signal"
    side = [p for n, p in named.items() if not n.startswith("base.")]
    assert side and all(p.grad is None or p.grad.abs().sum() == 0 for p in side), "decision layers were used"
    vocab = m.base.lm_head.base.weight
    assert not vocab.requires_grad and vocab.grad is None


def test_cot_loss_is_the_mean_over_sequences_of_the_mean_token_cross_entropy():
    from sors.training.cot import cot_batch, cot_lm_loss
    m, tok, d, _ = tiny_model(kind="minimal", feedback=True, trainable="full")
    m.eval()
    full = cot_batch(exs(), tok, d, "context-first", 512, False, False, paired=False)
    one = [cot_batch([e], tok, d, "context-first", 512, False, False, paired=False) for e in (exs()[0], exs()[2])]
    with torch.no_grad():
        got = cot_lm_loss(m, full)
        want = sum(cot_lm_loss(m, x) for x in one) / 2
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_training_with_cot_weight_logs_the_cot_loss_and_zero_weight_adds_nothing(capsys):
    from sors.training.loop import TrainConfig, train
    from sors.training.loop import _prepare_batch
    m, tok, d, _ = tiny_model(kind="minimal", feedback=True, grad_ckpt=True, trainable="full")
    spec0 = (list(d), 4, "context-first", 512, False, False, "minimal")
    plain, *_ = _prepare_batch(spec0, (exs(), None), tok)
    assert not any(k.startswith("cot_") for k in plain), "cot tensors must only appear when the loss is on"
    with_cot, *_ = _prepare_batch((*spec0, 1.0, False), (exs(), None), tok)
    assert "cot_input_ids" in with_cot
    cfg = TrainConfig(steps=2, batch_size=3, k_max=4, loss="menu", log_every=1, eval_every=10, initial_eval=False,
                      accumulate_gradients=True, micro_tokens=4096, cot_weight=1.0, data_workers=0)
    train(m, tok, d, lambda n, rng: exs()[:n], {}, cfg)
    out = capsys.readouterr().out
    assert "  cot " in out and "  cot_key " in out and "  cot_rest " in out, out


KEYED = ("Adding them: 12.25 + 5.25 = 17.50, then 17.50 + 7.50 = 25.00.\n"
         "Then 0.92 x 0.92 ≈ 0.85, and 0.8464, rounded to 0.85.\n"
         "So the value is 25.00, and 25.00 >= 25.00. The condition holds.\n"
         "That makes it yes.")


def test_key_spans_are_results_comparisons_verdicts_and_the_chosen_option():
    from sors.training.cot import key_spans
    got = [KEYED[a:b] for a, b in key_spans(KEYED)]
    assert got == ["17.50", "25.00", "0.85", "0.85", ">=", "holds", "yes"], got
    other = "It does not hold.\nSo I choose level 2."
    assert [other[a:b] for a, b in key_spans(other)] == ["does not hold", "2"]


def test_cot_batch_marks_the_label_tokens_that_overlap_a_key_span():
    from sors.training.cot import cot_batch
    m, tok, d, _ = tiny_model(kind="minimal", feedback=True)
    ex = MenuExample("red context", [10, 20], 1, 20, ["red", "blue"], question="Which color?", cot=KEYED)
    b = cot_batch([ex], tok, d, "context-first", 512, False, False, paired=False)
    key, labels = b["cot_key"][0], b["cot_labels"][0]
    assert key.shape == labels.shape and key.dtype == torch.bool
    assert not key[labels == -100].any(), "only working tokens can be key tokens"
    work = " " + KEYED
    offs = tok(work, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    first = int((labels != -100).nonzero()[0])  # 计算过程的第一个 token
    marked = [work[a:e] for (a, e), k in zip(offs, key[first:first + len(offs)].tolist()) if k]
    text = " ".join(marked)
    for s in ("17", "50", "25", "00", "85", ">", "holds", "yes"):
        assert s in marked or s in text, (s, marked)
    assert "Adding" not in marked and "condition" not in marked and "+" not in marked, marked


def test_cot_loss_splits_its_token_losses_into_key_and_rest():
    from sors.training.cot import cot_batch, cot_lm_loss
    m, tok, d, _ = tiny_model(kind="minimal", feedback=True, trainable="full")
    m.eval()
    ex = MenuExample("red context", [10, 20], 1, 20, ["red", "blue"], question="Which color?", cot=KEYED)
    b = cot_batch([ex], tok, d, "context-first", 512, False, False, paired=False)
    stats = {}
    with torch.no_grad():
        loss = cot_lm_loss(m, dict(b), stats)
    (ks, kn), (rs, rn) = stats["key"], stats["rest"]
    n = int((b["cot_labels"][0] != -100).sum())
    assert kn == int(b["cot_key"][0].sum()) and kn + rn == n and kn > 0 and rn > 0
    torch.testing.assert_close(torch.tensor((ks + rs) / n), loss.float().cpu(), atol=1e-4, rtol=1e-4)
