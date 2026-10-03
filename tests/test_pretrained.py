"""Complete inference exports carry every parameter and load without a base repo."""

import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from _runner import run
from test_decision import batch, tiny_model, tokenizer, tiny_backbone
from sors.core.decision import DecisionConfig, architecture_config
from sors.core.model import prepare_model


def api():
    assert importlib.util.find_spec("sors.core.pretrained") is not None, "complete-model export is missing"
    return importlib.import_module("sors.core.pretrained")


def settings():
    return {"layout": "context-first", "type_marker": False, "context_marker": False,
            "max_length": 8192, "candidate_prefix_cache": "off"}


def test_complete_weights_restore_frozen_parameters_and_decisions_without_a_base():
    module = api()
    for family in ("qwen3", "qwen35"):
        for kind in ("minimal", "structural", "candidate"):
            model, tok, d_ids, _ = tiny_model(kind, False, family=family, trainable="full")
            model.eval()
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(torch.randn_like(parameter) * 0.01)
            original = {name: value.clone() for name, value in model.state_dict().items()}
            data = batch(model, tok, d_ids)
            with torch.inference_mode():
                expected = model.forward_batch(data)
            with tempfile.TemporaryDirectory() as directory:
                folder = Path(directory) / "release"
                module.save_pretrained(model, tok, settings(), folder)
                assert len(list(folder.glob("*.safetensors"))) == 1
                assert not (folder / "backbone").exists()
                assert not (folder / "trained.safetensors").exists()
                assert not list(folder.rglob("*.py"))
                torch.manual_seed(819)
                with patch("transformers.AutoModelForCausalLM.from_pretrained",
                           side_effect=AssertionError("export tried to load base weights")):
                    restored, restored_tok, got_ids, config = module.load_pretrained(folder, device="cpu")
                assert got_ids == d_ids
                assert restored_tok.get_vocab() == tok.get_vocab()
                assert architecture_config(restored) == architecture_config(model)
                assert config["layout"] == "context-first"
                for name, value in original.items():
                    torch.testing.assert_close(restored.state_dict()[name], value, atol=0, rtol=0)
                    torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
                with torch.inference_mode():
                    torch.testing.assert_close(restored.forward_batch(data), expected, atol=0, rtol=0)


def test_tied_weights_are_stored_once_and_mixed_precision_survives_reload():
    module = api()
    from transformers import AutoModelForCausalLM
    tok, d_ids, ids = tokenizer()
    config = tiny_backbone(tok).config
    config.tie_word_embeddings = True
    raw = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16, attn_implementation="sdpa")
    model = prepare_model(raw, ids, 4, 8, 0., trainable="full",
                          decision=DecisionConfig(kind="minimal", dim=16, heads=4, blocks=2)).eval()
    with torch.no_grad():
        model.get_input_embeddings().rows.normal_(std=0.2)
    data = batch(model, tok, d_ids)
    with torch.inference_mode():
        expected = model.forward_batch(data)
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory) / "release"
        module.save_pretrained(model, tok, settings(), folder)
        with safe_open(folder / "model.safetensors", framework="pt") as stream:
            stored_elements = sum(stream.get_tensor(name).numel() for name in stream.keys())
        assert stored_elements == sum(p.numel() for p in model.parameters()), "shared weights were duplicated"
        restored, _, _, _ = module.load_pretrained(folder, device="cpu")
        assert restored.get_input_embeddings().base.weight.dtype == torch.bfloat16
        assert restored.option_proj.weight.dtype == torch.float32
        assert restored.blocks[0].read.in_proj_weight.dtype == torch.float32
        assert restored.base.lm_head.base.weight is restored.get_input_embeddings().base.weight
        assert restored.base.lm_head.emb.rows is restored.get_input_embeddings().rows
        assert restored.decoder.rotary_emb.inv_freq.dtype == torch.float32
        with torch.inference_mode():
            torch.testing.assert_close(restored.forward_batch(data), expected, atol=0, rtol=0)


def test_export_load_fails_when_any_backbone_or_decision_parameter_is_missing():
    module = api()
    model, tok, _, _ = tiny_model("minimal", False, trainable="full")
    for component in ("base.model.layers.0", "blocks.0"):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "release"
            module.save_pretrained(model, tok, settings(), folder)
            path = folder / "model.safetensors"
            with safe_open(path, framework="pt") as stream:
                metadata = stream.metadata()
            weights = load_file(path)
            name = next(name for name in weights if name.startswith(component))
            del weights[name]
            save_file(weights, path, metadata=metadata)
            try:
                module.load_pretrained(folder, device="cpu")
            except (RuntimeError, ValueError) as exc:
                assert "missing" in str(exc).lower(), str(exc)
            else:
                raise AssertionError("an incomplete model retained random parameters")


def test_export_refuses_overwriting_and_rejects_mismatched_token_ids():
    module = api()
    model, tok, _, _ = tiny_model("minimal")
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory) / "release"
        folder.mkdir()
        (folder / "keep.txt").write_text("existing data")
        try:
            module.save_pretrained(model, tok, settings(), folder)
        except (FileExistsError, ValueError):
            pass
        else:
            raise AssertionError("an existing directory was overwritten")
        assert (folder / "keep.txt").read_text() == "existing data"
        assert list(folder.iterdir()) == [folder / "keep.txt"]
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory) / "release"
        module.save_pretrained(model, tok, settings(), folder)
        path = folder / "config.json"
        config = json.loads(path.read_text())
        config["trainable_token_ids"][0], config["trainable_token_ids"][1] = (
            config["trainable_token_ids"][1], config["trainable_token_ids"][0])
        path.write_text(json.dumps(config))
        try:
            module.load_pretrained(folder, device="cpu")
        except ValueError as exc:
            assert "token" in str(exc).lower(), str(exc)
        else:
            raise AssertionError("a tokenizer/embedding coordinate mismatch was accepted")


def test_public_engine_loads_complete_directory_and_matches_reference_probabilities():
    module = api()
    from sors.serve.api import Choice
    from sors.serve.engine import Engine, load_engine

    model, tok, d_ids, _ = tiny_model("minimal", trainable="full")
    questions = {"color": Choice(type="choice", instructions="Which color?",
                                  criteria={"red": None, "blue": None})}
    reference = Engine(model, tok, d_ids).evaluate("red context", questions)
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory) / "release"
        module.save_pretrained(model, tok, settings(), folder)
        with patch("sors.core.attention.load_causal_lm", side_effect=AssertionError("base model was fetched")):
            try:
                engine = load_engine(folder, device="cpu", local_files_only=True)
            except TypeError as exc:
                raise AssertionError("complete directory loading still requires a separate base model") from exc
        actual = engine.evaluate("red context", questions)
        assert actual.probs == reference.probs
        assert actual.input_tokens == reference.input_tokens
        try:
            load_engine(folder, "an/unrelated-base", device="cpu")
        except ValueError as exc:
            assert "base" in str(exc).lower()
        else:
            raise AssertionError("an extra base model was silently ignored")


def test_complete_slot_model_preserves_the_language_model_readout():
    module = api()
    from sors.core.batch import collate
    from sors.core.model import last_logits
    from test_decision import examples

    tok, d_ids, ids = tokenizer()
    model = prepare_model(tiny_backbone(tok), ids, 4, 8, 0., trainable="full").eval()
    data = collate(examples(), tok, d_ids, 4)
    with torch.inference_mode():
        expected = last_logits(model, data["input_ids"], data["attention_mask"])
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory) / "release"
        module.save_pretrained(model, tok, settings(), folder)
        restored, _, _, _ = module.load_pretrained(folder, device="cpu")
        with torch.inference_mode():
            actual = last_logits(restored, data["input_ids"], data["attention_mask"])
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    run(globals())
