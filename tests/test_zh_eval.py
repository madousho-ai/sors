"""sors.data.zh_eval 的测试: 外部数据仓库的 zh-eval (中文阅读评测集, JevBench 格式) 读成按切片分开的评估集.

跑:  .venv/bin/pytest tests/test_zh_eval.py
"""

import argparse
import importlib.util
import json
import pathlib
import tempfile

from sors.data.jevbench import to_eval_example
from sors.data.zh_eval import FILE, load_zh_eval


def _row(i, slice_, surface="no", **extra):
    prov = {"rationale": "x", "surface_answer": surface, "phenomenon": "p", "slice": slice_, **extra}
    return {"id": f"zh-{slice_}-{i:03d}", "family": "intent", "state": "能退吗",
            "question": {"type": "noul", "instructions": "顾客是在要求退款吗？",
                         "criteria": {"true": "要求退款", "false": "没有要求退款"}},
            "labels": ["no", "yes"], "expected": "yes", "split": "private", "group": f"g{i}", "provenance": prov}


def _datasets(rows):
    d = pathlib.Path(tempfile.mkdtemp())
    (d / FILE).parent.mkdir(parents=True)
    (d / FILE).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return d


def test_one_set_per_slice_with_z9_split_by_direction_and_zp_by_language():
    rows = [_row(1, "z1"), _row(2, "z6"), _row(3, "z9", direction="zh_state_en_q"),
            _row(4, "z9", direction="en_state_zh_q"), _row(5, "zp", lang="zh"), _row(6, "zp", lang="en"),
            _row(7, "z1")]
    sets = load_zh_eval(datasets_dir=_datasets(rows))
    assert list(sets) == ["zh_z1", "zh_z6", "zh_z9_zhstate", "zh_z9_enstate", "zh_zp_zh", "zh_zp_en"]
    assert [len(v) for v in sets.values()] == [2, 1, 1, 1, 1, 1]


def test_each_item_is_the_menu_the_server_builds_with_the_decoy():
    r = _row(1, "z6")
    ex = load_zh_eval(datasets_dir=_datasets([r]))["zh_z6"][0]
    want = to_eval_example(r)
    assert (ex.query, ex.options, ex.gold_idx, ex.decoy) == (want.query, want.options, want.gold_idx, want.decoy)
    assert ex.decoy is not None and ex.decoy != ex.gold_idx


def test_train_eval_selector_expands_zheval_and_quarters_the_long_policy_batch():
    spec = importlib.util.spec_from_file_location("train_cli_zh", pathlib.Path(__file__).resolve().parent.parent / "scripts" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    d = _datasets([_row(1, "z1"), _row(2, "z8")])
    sets = mod.build_eval_sets(argparse.Namespace(eval="zheval", eval_batch_size=16, datasets_dir=str(d)))
    assert list(sets) == ["zh_z1", "zh_z8"]
    assert sets["zh_z8"].batch_size == 4 and sets["zh_z1"].batch_size == 16
