"""Exercise selection of a flat external checkout through the training CLI."""
import json
import os
import pathlib
import random
import shutil
import subprocess
import sys
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


def test_v5_overlap_check_uses_the_running_code_checkouts_cache():
    text = "violet lanterns illuminate quiet corridors beneath ancient stone arches"
    bank = [{"id": "pick", "ask": ["Which?"], "options": ["first", "second"]}]
    contexts = [{"id": "telecom_a", "label": "", "goal": "breadth", "text": text,
                 "questions": [{"question": "pick", "answer": "first"}]}]
    fixture = _data(telecom=(bank, contexts))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            code = pathlib.Path(tmp) / "custom-code-name"
            shutil.copytree(pathlib.Path(__file__).resolve().parents[1] / "src", code / "src",
                            ignore=shutil.ignore_patterns("__pycache__"))
            cache = code / "data/jevbench"
            cache.mkdir(parents=True)
            (cache / "easy.jsonl").write_text(json.dumps({"state": text}) + "\n")
            script = '''import sys
from decidophobia.data.synth_v5 import load_synth_v5
try:
    load_synth_v5(sys.argv[1])
except ValueError as exc:
    assert "JevBench" in str(exc), str(exc)
else:
    raise AssertionError("overlap was silently accepted under a custom checkout name")
'''
            env = {**os.environ, "PYTHONPATH": str(code / "src")}
            env.pop("JEVBENCH_DIR", None)
            result = subprocess.run([sys.executable, "-B", "-c", script, str(fixture)], cwd=code,
                                    env=env, capture_output=True)
            assert result.returncode == 0, result.stderr.decode()
    finally:
        shutil.rmtree(fixture)


if __name__ == "__main__":
    run(globals())
