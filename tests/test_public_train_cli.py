"""The real train CLI reads local official fixtures; no network, model weights or GPU."""

import collections
import importlib.util
import json
import pathlib
import random
import tempfile

from _runner import run
from test_public_decisions import _contract, _tools, _web, _conditional_rows
from sors.core.menu import row_alignment

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


def test_dataset_weights_control_real_draws_before_pairing():
    with tempfile.TemporaryDirectory() as d:
        manifest = _all_sources(pathlib.Path(d))
        args = _args(manifest, "contractnli+sharc+toolace", "--batch-size", "36",
                     "--dataset-weights", "contractnli=27,sharc=6,toolace=3",
                     "--consistency", "1", "--random-codes", "1")
        sample, _, info = _mod.build_data(args)
        batch = sample(36, random.Random(2))
        assert len(batch) == 72
        counts = collections.Counter(ex.context_label for ex in batch[::2])
        assert counts["Contract"] == 27, counts
        assert sorted(counts.values()) == [3, 6, 27], counts
        assert info["dataset_batch"] == {"contractnli": 27, "sharc": 6, "toolace": 3}
        for a, b in zip(batch[::2], batch[1::2]):
            assert a.option_names[a.gold_idx] == b.option_names[b.gold_idx]
            assert row_alignment(a, b)


def test_weighted_draws_keep_exact_half_synth_with_seven_public_sources():
    assert hasattr(_mod, "dataset_parts"), "dataset-level batch allocation is missing"
    names = ["synth-v5", "sharc", "toolace", "contractnli", "maud", "quality", "reclor", "logiqa2"]
    weights = _mod.parse_dataset_weights(
        "synth-v5.1=18,sharc=3,toolace=2,contractnli=4,maud=2,quality=3,reclor=2,logiqa2=2", names)
    assert _mod.dataset_parts(36, names, weights) == [18, 3, 2, 4, 2, 3, 2, 2]


def test_weighted_remainders_and_legacy_two_sampler_dataset_preserve_batch_size():
    assert hasattr(_mod, "dataset_parts"), "dataset-level batch allocation is missing"
    weights = _mod.parse_dataset_weights("synth=3,sharc=1", ["synth", "sharc"])
    assert _mod.dataset_parts(10, ["synth", "synth", "sharc"], weights) == [4, 4, 2]
    assert _mod.dataset_parts(10, ["synth", "synth", "sharc"], None) == [4, 3, 3]
    assert _mod.dataset_parts(0, ["synth", "synth", "sharc"], weights) == [0, 0, 0]


def test_dataset_weights_reject_missing_repeated_unknown_or_nonpositive_values():
    assert hasattr(_mod, "parse_dataset_weights"), "dataset weight validation is missing"
    for spec in ("sharc=1", "sharc=1,toolace=1,other=2", "sharc=1,toolace=0",
                 "sharc=1,toolace=-1", "sharc=1,toolace=nan", "sharc=1,toolace=inf",
                 "sharc=1,sharc=2,toolace=1", "sharc,toolace=1", ""):
        try:
            _mod.parse_dataset_weights(spec, ["sharc", "toolace"])
        except ValueError:
            continue
        raise AssertionError(f"invalid weights accepted: {spec}")
    try:
        _mod.parse_dataset_weights("synth-v5=1,synth-v5.1=1", ["synth-v5"])
    except ValueError:
        return
    raise AssertionError("duplicate alias weights accepted")


if __name__ == "__main__":
    run(globals())
