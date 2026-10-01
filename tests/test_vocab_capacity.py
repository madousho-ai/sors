"""Added slot tokens may exceed a base model's spare vocabulary rows. CPU only."""

import tempfile
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from _runner import run
from decidophobia.core.checkpoint import prepare_from_checkpoint, save_trained
from decidophobia.core.model import prepare_model
from decidophobia.training.loop import TrainConfig


def _base(tied=False):
    torch.manual_seed(7)
    return Qwen3ForCausalLM(Qwen3Config(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        tie_word_embeddings=tied, use_cache=False,
    ))


def _prepare(lm, ids):
    try:
        return prepare_model(lm, ids, None, None, 0.0, trainable="full")
    except IndexError as exc:
        raise AssertionError("slot tokens must fit after preparing the model") from exc


def test_slots_beyond_base_vocab_receive_finite_logits_and_gradients():
    for tied in (False, True):
        lm = _base(tied)
        original = lm.get_input_embeddings().weight.detach().clone()
        original_head = lm.lm_head.weight.detach().clone()
        m = _prepare(lm, list(range(60, 68)))
        emb = m.get_input_embeddings()
        assert emb.base.weight.shape[0] == m.lm_head.base.weight.shape[0] == 68
        assert m.config.vocab_size == 68
        assert torch.equal(emb.base.weight[:64], original)
        assert torch.equal(m.lm_head.base.weight[:64], original_head)
        assert not emb.base.weight.requires_grad and not m.lm_head.base.weight.requires_grad
        assert m.lm_head.emb.rows is emb.rows
        logits = m(input_ids=torch.tensor([[1, 65, 2, 67]])).logits
        assert logits.shape == (1, 4, 68) and torch.isfinite(logits).all()
        torch.nn.functional.cross_entropy(logits[:, -1], torch.tensor([66])).backward()
        assert torch.isfinite(emb.rows.grad).all() and emb.rows.grad.abs().sum() > 0
        assert m.model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
        assert emb.base.weight.grad is None and m.lm_head.base.weight.grad is None


def test_existing_spare_vocab_is_preserved_without_resizing():
    lm = _base()
    embedding, head = lm.get_input_embeddings().weight, lm.lm_head.weight
    m = _prepare(lm, list(range(40, 48)))
    assert m.get_input_embeddings().base.weight is embedding
    assert m.lm_head.base.weight is head
    assert m.config.vocab_size == 64
    assert m(input_ids=torch.tensor([[1, 41, 2]])).logits.shape[-1] == 64


def test_expanded_checkpoint_reloads_with_a_different_initialization_seed():
    for tied in (False, True):
        # Row 64 is deliberately a gap: frozen added rows must also reproduce.
        ids = [60, 65, 66, 67]
        m = _prepare(_base(tied), ids).eval()
        with torch.no_grad():
            for p in m.parameters():
                if p.requires_grad:
                    p.add_(0.003)
        probe = torch.tensor([[1, 64, 65, 2]])
        want = m(input_ids=probe).logits.detach()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trained.safetensors"
            save_trained(m, ids, TrainConfig(), path)
            fresh = _base(tied)
            torch.manual_seed(987)
            restored, _ = prepare_from_checkpoint(fresh, ids, path)
            assert torch.equal(restored.eval()(input_ids=probe).logits, want)


if __name__ == "__main__":
    run(globals())
