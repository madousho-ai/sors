"""Shared prefixes must preserve joint inputs, branch isolation and gradients."""

from dataclasses import replace
from contextlib import contextmanager

import torch

from _runner import run
from test_candidate import candidate_batch, candidate_model, examples
from decidophobia.core.menu import reorder_menu


def set_mode(m, mode):
    assert hasattr(m, "set_candidate_prefix_cache"), "candidate shared-prefix path is missing"
    m.set_candidate_prefix_cache(mode)


@contextmanager
def trace(m):
    calls = []
    def before(module, args, kwargs):
        cache = kwargs.get("past_key_values")
        calls.append({"ids": kwargs["input_ids"].detach().cpu().clone(),
                      "past": int(cache.get_seq_length()) if cache is not None else 0,
                      "cache": bool(kwargs.get("use_cache")),
                      "mask": kwargs["attention_mask"].detach().cpu().clone()})
    handle = m.decoder.register_forward_pre_hook(before, with_kwargs=True)
    try:
        yield calls
    finally:
        handle.remove()


def test_prefix_and_suffix_reconstruct_every_original_joint_input():
    _, tok, d, _ = candidate_model(0)
    data = candidate_batch(tok, d)
    assert "prefix_lengths" in data, "shared-prefix token boundaries are missing"
    for row, ex in enumerate(examples()):
        prefix = data["input_ids"][row][data["attention_mask"][row].bool()][:data["prefix_lengths"][row]].tolist()
        for col in range(len(ex.options)):
            suffix = data["suffix_input_ids"][row, col][data["suffix_attention_mask"][row, col].bool()].tolist()
            full = data["option_input_ids"][row, col][data["option_attention_mask"][row, col].bool()].tolist()
            assert prefix and suffix and prefix + suffix == full


def test_prefix_split_handles_a_real_bpe_merge_across_the_textual_boundary():
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast
    from decidophobia.core.tokens import install_d_tokens
    vocab = {s: i for i, s in enumerate(["[PAD]", "[UNK]", "\n"] + [chr(i) for i in range(32, 127)])}
    merges = []
    previous = "\n"
    for char in "\nOption":
        merges.append((previous, char))
        previous += char
        vocab[previous] = len(vocab)
    backend = Tokenizer(models.BPE(vocab, merges, unk_token="[UNK]"))
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]")
    d = install_d_tokens(tok)
    data = candidate_batch(tok, d, [examples()[0]], max_length=1024)
    assert "prefix_lengths" in data, "shared-prefix token boundaries are missing"
    length = int(data["prefix_lengths"][0])
    assert 0 < length < int(data["attention_mask"][0].sum()), "boundary merge was not detected"
    prefix = data["input_ids"][0][data["attention_mask"][0].bool()][:length].tolist()
    for col in range(3):
        suffix = data["suffix_input_ids"][0, col][data["suffix_attention_mask"][0, col].bool()].tolist()
        full = data["option_input_ids"][0, col][data["option_attention_mask"][0, col].bool()].tolist()
        assert prefix + suffix == full


def test_candidate_default_prefills_once_and_processes_only_suffixes_afterwards():
    for family in ("qwen3", "qwen35"):
        m, tok, d, _ = candidate_model(0, family)
        assert getattr(m, "candidate_prefix_cache", None) == "auto", "new candidate models must share prefixes by default"
        data = candidate_batch(tok, d, [examples()[0]])
        with trace(m) as calls:
            m.eval().forward_batch(data)
        prefills = [c for c in calls if c["cache"] and c["past"] == 0]
        assert len(prefills) == 1 and prefills[0]["ids"].shape[0] == 1
        length = int(data["prefix_lengths"][0])
        assert prefills[0]["ids"].shape[1] == length
        assert all(c["past"] == length for c in calls[1:])
        processed = sum(c["ids"].numel() for c in calls)
        assert processed == length + int(data["suffix_attention_mask"].sum())
        assert processed < int(data["option_attention_mask"].sum())


def test_identical_question_views_share_one_prefix_and_keep_candidate_order():
    m, tok, d, _ = candidate_model(2)
    set_mode(m, "on")
    ex = examples()[0]
    other = reorder_menu(ex, [2, 0, 1], [210, 99, 13])
    data = candidate_batch(tok, d, [ex, other])
    with trace(m) as calls:
        logits = m.eval().forward_batch(data)
    assert len([c for c in calls if c["cache"] and c["past"] == 0]) == 1
    p = logits[0, data["slot_ids"][0, :3]].softmax(-1)
    q = logits[1, data["slot_ids"][1, :3]].softmax(-1)
    torch.testing.assert_close(q, p[[2, 0, 1]], atol=2e-6, rtol=2e-5)


def test_mixed_length_suffixes_match_complete_forward_without_padding_the_recurrent_stream():
    for family in ("qwen3", "qwen35"):
        m, tok, d, _ = candidate_model(2, family)
        ex = replace(examples()[0], option_names=["red", "long blue", "long long green"])
        data = candidate_batch(tok, d, [ex])
        m.eval()
        set_mode(m, "off")
        full = m.forward_batch(data)
        set_mode(m, "on")
        with trace(m) as calls:
            cached = m.forward_batch(data)
        torch.testing.assert_close(cached, full, atol=3e-5, rtol=3e-4)
        assert all(c["mask"].all() for c in calls), "padding advanced a recurrent branch state"
        torch.testing.assert_close(m.forward_batch(data), cached, atol=0, rtol=0)


def test_frozen_training_keeps_prefill_outside_checkpoint_recomputation():
    m, tok, d, _ = candidate_model(2, checkpointed=True)
    set_mode(m, "on")
    m.train()
    data = candidate_batch(tok, d, [examples()[0]])
    with trace(m) as calls:
        loss = -m.forward_batch(data).log_softmax(-1)[:, d[0]].mean()
        loss.backward()
    assert len([c for c in calls if c["cache"] and c["past"] == 0]) == 1
    assert m.scorer.weight.grad.abs().sum() > 0
    assert m.blocks[0].attention.in_proj_weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in m.base.parameters())
    assert m.decoder.training, "encoding mode leaked into the next model call"


def test_auto_preserves_trainable_backbone_and_marker_gradients():
    for trainable, marker in (("full", False), ("attn", False), ("decision-only", True)):
        m, tok, d, _ = candidate_model(0, trainable=trainable)
        set_mode(m, "auto")
        m.train()
        data = candidate_batch(tok, d, [examples()[0]], type_marker=marker)
        with trace(m) as calls:
            (-m.forward_batch(data).log_softmax(-1)[:, d[0]].mean()).backward()
        assert all(not c["cache"] for c in calls)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.base.parameters())
        set_mode(m, "on")
        try:
            m.forward_batch(data)
        except ValueError as exc:
            assert "trainable" in str(exc)
        else:
            raise AssertionError("forced cache silently cut encoder gradients")


def test_no_grad_inference_can_cache_a_finetuned_model():
    m, tok, d, _ = candidate_model(0, trainable="full")
    set_mode(m, "auto")
    with torch.no_grad(), trace(m) as calls:
        m.eval().forward_batch(candidate_batch(tok, d, [examples()[0]]))
    assert len([c for c in calls if c["cache"] and c["past"] == 0]) == 1


def cache_tensors(cache):
    result = {}
    for i, layer in enumerate(cache.layers):
        for key in ("keys", "values", "conv_states", "recurrent_states"):
            value = getattr(layer, key, None)
            values = value.items() if isinstance(value, dict) else [(0, value)]
            for index, tensor in values:
                if isinstance(tensor, torch.Tensor):
                    result[(i, key, index)] = tensor
    return result


def test_hybrid_fork_isolates_kv_convolution_and_recurrent_state():
    from transformers import DynamicCache
    from decidophobia.core.candidate_cache import fork_cache
    m, _, _, _ = candidate_model(0, "qwen35")
    m.eval()
    ids = torch.tensor([[2, 3, 4, 5, 6, 7]])
    with torch.no_grad():
        cache = m.decoder(input_ids=ids, attention_mask=torch.ones_like(ids),
                          past_key_values=DynamicCache(config=m.config), use_cache=True).past_key_values
        before = {k: v.clone() for k, v in cache_tensors(cache).items()}
        assert {k[1] for k in before} == {"keys", "values", "conv_states", "recurrent_states"}
        branch = fork_cache(cache, 2, ids.device)
        for key, tensor in cache_tensors(branch).items():
            torch.testing.assert_close(tensor, before[key].repeat_interleave(2, dim=0), atol=0, rtol=0)
            tensor[0].add_(1)
            torch.testing.assert_close(tensor[1], before[key][0], atol=0, rtol=0)
        for key, tensor in cache_tensors(cache).items():
            torch.testing.assert_close(tensor, before[key], atol=0, rtol=0)
        assert cache.get_seq_length() == branch.get_seq_length() == 6


def test_one_token_suffix_uses_the_same_hybrid_state_as_full_forward():
    from transformers import DynamicCache
    from decidophobia.core.candidate_cache import fork_cache
    m, _, _, _ = candidate_model(0, "qwen35")
    m.eval()
    prefix = torch.tensor([[2, 3, 4, 5, 6]])
    suffix = torch.tensor([[7], [8]])
    with torch.no_grad():
        cache = m.decoder(input_ids=prefix, attention_mask=torch.ones_like(prefix),
                          past_key_values=DynamicCache(config=m.config), use_cache=True).past_key_values
        fork = fork_cache(cache, 2, prefix.device)
        cached = m.decoder(input_ids=suffix, attention_mask=torch.ones(2, 6, dtype=torch.long),
                           position_ids=torch.full((2, 1), 5), past_key_values=fork, use_cache=True).last_hidden_state[:, -1]
        full = torch.cat((prefix.expand(2, -1), suffix), dim=1)
        want = m.decoder(input_ids=full, attention_mask=torch.ones_like(full), use_cache=False).last_hidden_state[:, -1]
    torch.testing.assert_close(cached, want, atol=3e-5, rtol=3e-4)


if __name__ == "__main__":
    run(globals())
