"""External dataset roots are explicit, independent of caches and cwd."""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from unittest.mock import patch

from _runner import run
from decidophobia.data import label_names as names_module


def test_description_labels_read_the_selected_asset_root():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        (root / "label-descriptions").mkdir()
        (root / "label-descriptions/banking77.json").write_text(json.dumps({"close": "Closing the account"}))
        got = names_module.label_names("banking77", ["close"], "desc", datasets_dir=root)
        assert got == {0: "close: Closing the account"}


def test_raw_labels_work_without_an_asset_checkout():
    got = names_module.label_names("banking77", ["close"], "raw", datasets_dir="/missing/dataset-checkout")
    assert got == {0: "close"}


def test_synthetic_loaders_resolve_the_root_at_call_time():
    from decidophobia.data.synth import load_synth, load_synth_binary
    from decidophobia.data.simple_eval import load_simple_eval
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        (root / "synth-intents-v2.5").mkdir()
        (root / "synth-intents-v2.5/toy.jsonl").write_text(json.dumps({
            "id": "close", "domain": "toy", "description": "Close account", "utterances": ["close it"],
            "question": "Close?", "answers": [True]}) + "\n")
        (root / "synth-simple-eval").mkdir()
        (root / "synth-simple-eval/synth-simple-eval.jsonl").write_text(json.dumps({
            "context": "yes", "question": "Proceed?", "qtype": "bool", "options": ["no", "yes"], "answer": 1}) + "\n")
        with patch.dict(os.environ, {"DECIDOPHOBIA_DATASETS_DIR": tmp}):
            data, domains = load_synth()
            assert data.queries == ["close it"] and domains == ["toy"]
            assert load_synth_binary().labels == [1]
            assert load_simple_eval()["simple_bool"][0].gold_idx == 1


def test_benchmark_adapters_forward_the_selected_description_root():
    from decidophobia.data.banking77 import load_banking77
    from decidophobia.data.massive import load_massive
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        (root / "label-descriptions").mkdir()
        for dataset in ("banking77", "massive"):
            (root / f"label-descriptions/{dataset}.json").write_text(json.dumps({"close": "Close it"}))
        with patch("decidophobia.data.banking77._fetch", return_value=[("please close", "close")]):
            assert load_banking77(labels="desc", datasets_dir=root)[0].names == {0: "close: Close it"}
        with patch("decidophobia.data.massive._read_rows", return_value=[
                {"intent": "close", "partition": "test", "utt": "please close"}]):
            assert load_massive(labels="desc", datasets_dir=root).names == {0: "close: Close it"}


def test_asset_configuration_resolves_cli_before_environment_and_reports_missing_files():
    from decidophobia.data.paths import add_datasets_argument, asset_path, datasets_root
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        root = pathlib.Path(a)
        (root / "synth-intents-v2.5").mkdir()
        with patch.dict(os.environ, {"DECIDOPHOBIA_DATASETS_DIR": b}):
            parser = argparse.ArgumentParser()
            add_datasets_argument(parser)
            args = parser.parse_args(["--datasets-dir", a])
            assert datasets_root(args.datasets_dir) == root
            assert datasets_root() == pathlib.Path(b)
            assert asset_path("synth-intents-v2.5", args.datasets_dir) == root / "synth-intents-v2.5"
            try:
                asset_path("synth-intents-v3", args.datasets_dir)
            except FileNotFoundError as exc:
                assert "--datasets-dir" in str(exc) and str(root) in str(exc)
            else:
                raise AssertionError("missing assets must fail with configuration guidance")


def test_first_import_allows_explicit_assets_despite_an_empty_environment():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        (root / "label-descriptions").mkdir()
        (root / "label-descriptions/banking77.json").write_text(json.dumps({"close": "Close it"}))
        script = '''import sys
from decidophobia.data.label_names import label_names
from decidophobia.data import synth, synth_v3, synth_v5, simple_eval
assert label_names("banking77", ["close"], "raw") == {0: "close"}
assert label_names("banking77", ["close"], "desc", datasets_dir=sys.argv[1]) == {0: "close: Close it"}
'''
        result = subprocess.run([sys.executable, "-B", "-c", script, tmp], capture_output=True,
                                env={**os.environ, "DECIDOPHOBIA_DATASETS_DIR": ""})
        assert result.returncode == 0, result.stderr.decode()


if __name__ == "__main__":
    run(globals())
