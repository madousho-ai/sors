"""sors-eval/ in the dataset repository is a held-out evaluation set. Training must never read it.

Run: PYTHONPATH=src .venv/bin/python tests/test_sors_eval_isolation.py
"""

import importlib.util
import pathlib

from _runner import run

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("train_cli", ROOT / "scripts" / "train.py")
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)


def test_no_training_dataset_name_selects_the_eval_set():
    for name in ("sors-eval", "sors_eval", "eval"):
        try:
            _mod.parse_datasets(name)
        except SystemExit:
            continue
        raise AssertionError(f"--dataset {name} was accepted")
    assert not any("eval" in n for n in _mod.KNOWN), _mod.KNOWN


def test_no_training_source_reads_the_eval_set():
    hits = [str(p.relative_to(ROOT)) for d in ("src", "scripts") for p in (ROOT / d).rglob("*.py")
            if "sors-eval" in p.read_text(encoding="utf-8")]
    assert hits == [], hits


if __name__ == "__main__":
    run(globals())
