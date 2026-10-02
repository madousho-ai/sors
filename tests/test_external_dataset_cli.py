"""Exercise selection of a flat external checkout through the training CLI."""
import json
import os
import pathlib
import random
import shutil
import tempfile
from unittest.mock import patch

from _runner import run
from test_train_cli import _mod
from test_synth_v5 import _data
from decidophobia.data.paths import datasets_root


def test_cli_uses_selected_assets_for_training_and_evaluation():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        fixture = _data()
        try:
            shutil.copytree(fixture, root / "synth-intents-v5.1")
        finally:
            shutil.rmtree(fixture)
        (root / "synth-simple-eval").mkdir()
        (root / "synth-simple-eval/synth-simple-eval.jsonl").write_text(json.dumps({
            "context": "local fixture", "question": "Proceed?", "qtype": "bool",
            "options": ["no", "yes"], "answer": 1}) + "\n")
        with patch.dict(os.environ, {"DECIDOPHOBIA_DATASETS_DIR": "/missing/environment-root"}):
            try:
                args = _mod.build_parser().parse_args([
                    "--dataset", "synth-v5.1", "--eval", "simple", "--datasets-dir", tmp])
            except SystemExit:
                raise AssertionError("the training CLI must accept --datasets-dir") from None
            sample, evaluation, info = _mod.build_data(args)
            assert info["synth_v5_items"] == 10
            assert len(sample(8, random.Random(7))) == 8
            assert evaluation["simple_bool"].examples[0].query == "local fixture"
            assert evaluation["simple_bool"].examples[0].gold_idx == 1


def test_v3_loader_uses_explicit_root_with_an_invalid_environment():
    from decidophobia.data.synth_v3 import load_synth_v3
    root = datasets_root()
    with patch.dict(os.environ, {"DECIDOPHOBIA_DATASETS_DIR": "/missing/environment-root"}):
        by_domain = load_synth_v3(datasets_dir=root)
    assert set(by_domain) == {"telecom", "hotel", "browser_agent", "sec_ops", "coding_ci"}
    assert sum(map(len, by_domain.values())) == 688


if __name__ == "__main__":
    run(globals())
