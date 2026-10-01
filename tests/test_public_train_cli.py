"""The real train CLI reads local official fixtures; no network, model weights or GPU."""

import collections
import importlib.util
import json
import pathlib
import random
import tempfile

from _runner import run
from test_public_decisions import _contract, _tools, _web, _conditional_rows
from decidophobia.core.menu import row_alignment

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts/train.py"
_spec = importlib.util.spec_from_file_location("public_train_cli", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _args(manifest, dataset, *extra):
    try:
        return _mod.build_parser().parse_args(["--dataset", dataset, "--public-manifest", str(manifest),
                                              "--eval", "simple", *extra])
    except SystemExit:
        raise AssertionError("train CLI has not connected the local public manifest") from None


def _all_sources(root):
    sources = []

    def add(name, raw, **kw):
        path = root / f"{name}.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        sources.append({"dataset": name, "split": "train", "path": path.name,
                        "source": "official schema fixture", "license": "fixture-only", **kw})

    add("contractnli", _contract())
    add("maud", [{"text": "Cash consideration.", "question": "Type of Consideration", "subquestion": "",
                  "answer": "All Cash", "label": 0}], catalog="catalog.json")
    (root / "catalog.json").write_text(json.dumps({"Type of Consideration": ["All Cash", "All Stock"]}))
    add("legalbench", [{"index": 0, "text": "An out-of-court statement.", "answer": "Yes"}], task="hearsay")
    add("sharc", [{"snippet": "Adults may apply.", "question": "Can I apply?", "scenario": "I am 20.",
                   "history": [], "evidence": [], "answer": "Yes"}])
    add("conditionalqa", _conditional_rows(), documents="documents.json")
    (root / "documents.json").write_text(json.dumps([{"url": "doc-a", "title": "Rules", "contents": ["Age 18."]}]))
    add("mind2web", [_web()])
    add("toolace", [_tools()])
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"version": 1, "sources": sources}))
    return manifest


def test_public_datasets_share_the_existing_batch_mixing_and_pairing_pipeline():
    with tempfile.TemporaryDirectory() as d:
        manifest = _all_sources(pathlib.Path(d))
        names = "contractnli+maud+legalbench+sharc+conditionalqa+mind2web+toolace"
        args = _args(manifest, names, "--consistency", "1", "--random-codes", "1")
        sample, evals, info = _mod.build_data(args)
        assert set(info["public_decisions"]) == set(names.split("+"))
        assert info["public_decisions"]["conditionalqa"]["items"] == 2
        assert info["public_decisions"]["mind2web"]["skipped"] == {"no_positive": 1}
        assert evals and all(e.codes is None for es in evals.values() for e in es.examples)
        batch = sample(14, random.Random(0))
        assert len(batch) == 28
        counts = collections.Counter(e.context_label for e in batch)
        assert set(counts.values()) == {4} and len(counts) == 7, counts
        for a, b in zip(batch[::2], batch[1::2]):
            assert [b.options[j] for j in row_alignment(a, b)] == a.options
            assert a.option_names[a.gold_idx] == b.option_names[b.gold_idx]
            assert a.codes is not None and b.codes is not None


def test_fixed_official_menus_reject_a_training_capacity_that_is_too_small():
    with tempfile.TemporaryDirectory() as d:
        manifest = _all_sources(pathlib.Path(d))
        args = _args(manifest, "contractnli", "--k-max", "2")
        try:
            _mod.build_data(args)
        except SystemExit as e:
            assert "k-max" in str(e), str(e)
            return
        raise AssertionError("three-choice official menus were accepted with training capacity two")


def test_old_dataset_defaults_and_sampling_are_unchanged():
    with tempfile.TemporaryDirectory() as d:
        manifest = _all_sources(pathlib.Path(d))
        # An unused manifest must not be opened by a legacy-only run.
        args = _args(str(manifest) + ".missing", "synth-v5.1")
        assert _mod.build_parser().parse_args([]).dataset == "synth"
        sample, _, info = _mod.build_data(args)
        assert "public_decisions" not in info
        assert len(sample(4, random.Random(0))) == 4


if __name__ == "__main__":
    run(globals())
