"""sors.data.json_eval 的测试: 外部数据仓库的 json-eval (JSON 阅读能力评测集, JevBench 格式) 读成按能力分开的评估集.

跑:  .venv/bin/pytest tests/test_json_eval.py
"""

import json
import pathlib
import tempfile

from sors.data.jevbench import to_eval_example
from sors.data.json_eval import FILE, load_json_eval


def _row(i, family, slice_, size=None, surface="no"):
    prov = {"rationale": "x", "surface_answer": surface, "slice": slice_}
    if size:
        prov["size"] = size
    return {"id": f"je-{slice_}-{i:03d}", "family": family, "state": {"a": {"b": "c"}},
            "question": {"type": "noul", "instructions": "Read only .a.b. Is it c?",
                         "criteria": {"true": "it is c", "false": "it is not c"}},
            "labels": ["no", "yes"], "expected": "yes", "split": "private", "group": f"g{i}", "provenance": prov}


def _datasets(rows):
    d = pathlib.Path(tempfile.mkdtemp())
    (d / FILE).parent.mkdir(parents=True)
    (d / FILE).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return d


def test_one_set_per_slice_and_c6_split_by_size():
    rows = [_row(1, "c1", "c1"), _row(2, "c2", "c2"), _row(3, "c3", "tr"),
            _row(4, "c6", "c6", "s"), _row(5, "c6", "c6", "l"), _row(6, "c6", "c6", "m"), _row(7, "c6", "c6", "s")]
    sets = load_json_eval(datasets_dir=_datasets(rows))
    assert list(sets) == ["json_c1", "json_c2", "json_tr", "json_c6_s", "json_c6_m", "json_c6_l"]
    assert [len(v) for v in sets.values()] == [1, 1, 1, 2, 1, 1]


def test_each_item_is_the_menu_the_server_builds_with_the_decoy():
    r = _row(1, "c2", "c2")
    ex = load_json_eval(datasets_dir=_datasets([r]))["json_c2"][0]
    want = to_eval_example(r)
    assert (ex.query, ex.options, ex.gold_idx, ex.decoy, ex.family) == \
           (want.query, want.options, want.gold_idx, want.decoy, want.family)
    assert ex.decoy is not None and ex.decoy != ex.gold_idx


def test_train_eval_selector_expands_jsoneval_into_its_sets():
    import argparse
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_cli_json", pathlib.Path(__file__).resolve().parent.parent / "scripts" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    d = _datasets([_row(1, "c1", "c1"), _row(2, "c6", "c6", "l")])
    sets = mod.build_eval_sets(argparse.Namespace(eval="jsoneval", eval_batch_size=16, datasets_dir=str(d)))
    assert list(sets) == ["json_c1", "json_c6_l"]
    assert sets["json_c6_l"].batch_size == 4 and sets["json_c1"].batch_size == 16
