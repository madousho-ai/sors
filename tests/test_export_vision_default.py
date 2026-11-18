"""export-model.py 默认把视觉编码器打进导出目录: 决定从哪个底座挂视觉编码器 (export_vision)."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/export-model.py"
_spec = importlib.util.spec_from_file_location("export_model", _SCRIPT)
export_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_model)

CANDIDATE = {"kind": "candidate", "dim": 16, "heads": 4, "blocks": 0, "layers": [], "feedback": False,
             "option_batch_size": 2}


def checkpoint(tmp_path, architecture=None) -> Path:
    from safetensors.torch import save_file
    path = tmp_path / "trained.safetensors"
    metadata = {} if architecture is None else {"architecture": json.dumps(architecture)}
    save_file({"d_embed": torch.zeros(1)}, str(path), metadata=metadata)
    return path


def complete(tmp_path, architecture, vision=False) -> Path:
    record = {"model_type": "sors", "architecture": architecture}
    if vision:
        record["vision"] = {"config": {}, "image_token_id": 1}
    (tmp_path / "config.json").write_text(json.dumps(record))
    return tmp_path


def test_a_candidate_checkpoint_takes_its_vision_encoder_from_the_base_model(tmp_path):
    assert export_model.export_vision(checkpoint(tmp_path, CANDIDATE), "Qwen/Base") == "Qwen/Base"


def test_an_explicit_vision_model_wins_over_the_base_model(tmp_path):
    got = export_model.export_vision(checkpoint(tmp_path, CANDIDATE), "Qwen/Base", vision_model="Qwen/Other")
    assert got == "Qwen/Other"


def test_no_vision_exports_text_only(tmp_path):
    assert export_model.export_vision(checkpoint(tmp_path, CANDIDATE), "Qwen/Base", no_vision=True) is None


def test_no_vision_and_a_vision_model_together_are_refused(tmp_path):
    with pytest.raises(ValueError, match="--no-vision"):
        export_model.export_vision(checkpoint(tmp_path, CANDIDATE), "Qwen/Base", vision_model="Qwen/Base",
                                   no_vision=True)


def test_models_that_cannot_read_images_export_without_vision(tmp_path):
    assert export_model.export_vision(checkpoint(tmp_path), "Qwen/Base") is None
    assert export_model.export_vision(complete(tmp_path, {"kind": "slots"}), None) is None


def test_a_complete_directory_that_already_carries_vision_keeps_it(tmp_path):
    assert export_model.export_vision(complete(tmp_path, CANDIDATE, vision=True), None) is None


def test_a_complete_candidate_directory_without_vision_needs_a_vision_model(tmp_path):
    directory = complete(tmp_path, CANDIDATE)
    with pytest.raises(ValueError, match="--vision-model"):
        export_model.export_vision(directory, None)
    assert export_model.export_vision(directory, None, vision_model="Qwen/Base") == "Qwen/Base"
