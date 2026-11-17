"""synth-intents-v6: v5.25 上的一份逐道绑定名单 (selection.json), 训练时读 v5.25 再按名单筛.

名单由 spec.json 的规则生成 (build_selection: ladder_split 留出每类的评测题, ladder_pool 每类取 n 道), 记下 v5.25
每个文件的 sha256; 读的时候 v5.25 对不上 sha256、或名单里有 v5.25 没有的绑定, 都报错, 不悄悄训一份别的数据.
"""
import hashlib
import json
import pathlib
import shutil
import tempfile

import pytest

from sors.data.synth_v5 import load_synth_v5
from sors.data.synth_v6 import build_selection, load_synth_v6, select_bindings, source_sha256
from test_synth_v5 import EDITED, TELECOM_C, TELECOM_Q, _data


def _key(it):
    return (it.domain, it.context_id, it.question_id)


def _assets(keys, sha=None):
    """一个数据根: synth-intents-v5.25 (带改动材料的小样本) 与 synth-intents-v6 (selection.json 选 keys)."""
    root = pathlib.Path(tempfile.mkdtemp())
    src = _data(telecom=(TELECOM_Q, [*TELECOM_C, EDITED]))
    shutil.copytree(src, root / "synth-intents-v5.25")
    shutil.rmtree(src)
    v6 = root / "synth-intents-v6"
    v6.mkdir()
    sel = {"source": "synth-intents-v5.25", "source_sha256": sha or source_sha256(root / "synth-intents-v5.25"),
           "spec": {"note": "fixture"}, "bindings": [list(k) for k in keys]}
    (v6 / "selection.json").write_text(json.dumps(sel))
    return root


def test_only_the_selected_bindings_are_loaded_in_source_order_with_edits_pointing_inside():
    full = load_synth_v5(_data(telecom=(TELECOM_Q, [*TELECOM_C, EDITED])))
    hotel = next(it for it in full if it.domain == "hotel")
    want = [("telecom", "telecom_queue_a/cf", "pickup"), ("hotel", hotel.context_id, hotel.question_id),
            ("telecom", "telecom_queue_a", "pickup")]
    root = _assets(want)
    try:
        items = load_synth_v6(datasets_dir=root)
    finally:
        shutil.rmtree(root)
    assert [_key(it) for it in items] == [k for k in map(_key, full) if k in set(want)]
    edited = next(it for it in items if it.context_id == "telecom_queue_a/cf")
    assert _key(items[edited.contrast]) == ("telecom", "telecom_queue_a", "pickup")


def test_an_edit_whose_source_was_not_selected_loses_the_link_instead_of_pointing_outside():
    pool = select_bindings(load_synth_v5(_data(telecom=(TELECOM_Q, [*TELECOM_C, EDITED]))),
                           [("telecom", "telecom_queue_a/cf", "pickup")])
    assert len(pool) == 1 and pool[0].contrast is None and pool[0].pair is None


def test_a_selected_binding_missing_from_the_source_is_an_error():
    root = _assets([("telecom", "telecom_queue_a", "pickup"), ("telecom", "nope", "pickup")])
    try:
        with pytest.raises(ValueError, match="nope"):
            load_synth_v6(datasets_dir=root)
    finally:
        shutil.rmtree(root)


def test_a_source_that_changed_since_the_selection_was_written_is_an_error():
    root = _assets([("telecom", "telecom_queue_a", "pickup")], sha={"telecom.contexts.json": "0" * 64})
    try:
        with pytest.raises(ValueError, match="sha256"):
            load_synth_v6(datasets_dir=root)
    finally:
        shutil.rmtree(root)


def test_source_sha256_covers_every_file_except_caches():
    d = pathlib.Path(tempfile.mkdtemp())
    try:
        (d / "a.json").write_text("{}")
        (d / "sub").mkdir()
        (d / "sub" / "b.json").write_text("[]")
        (d / "__pycache__").mkdir()
        (d / "__pycache__" / "x.pyc").write_bytes(b"\0")
        assert source_sha256(d) == {"a.json": hashlib.sha256(b"{}").hexdigest(),
                                    "sub/b.json": hashlib.sha256(b"[]").hexdigest()}
    finally:
        shutil.rmtree(d)


def test_build_selection_takes_n_per_class_after_holding_out_each_class_and_dropping_listed_materials():
    full = load_synth_v5(_data(telecom=(TELECOM_Q, [*TELECOM_C, EDITED])))
    spec = {"classes": {"base": ["base"]}, "n": {"base": 3}, "eval_n": 1, "salt": "t:"}
    keys = build_selection(full, spec, drop={"telecom_rule_1"})
    assert 3 <= len(keys) <= 4 and len(set(keys)) == len(keys)
    assert not any(c == "telecom_rule_1" for _, c, _ in keys)
    assert keys == build_selection(full, spec, drop={"telecom_rule_1"})


# ---- --dataset synth-v6

def test_synth_v6_is_its_own_selector_and_does_not_combine_with_v5():
    from test_train_cli import _mod
    assert _mod.parse_datasets("synth-v6") == ["synth-v6"]
    for both in ("synth-v6+synth-v5", "synth-v6+synth-v5.25"):
        with pytest.raises(SystemExit, match="synth-v6"):
            _mod.parse_datasets(both)


def test_training_on_synth_v6_reads_only_the_selection_and_records_it():
    import random
    from test_train_cli import _args, _mod
    full = load_synth_v5(_data(telecom=(TELECOM_Q, [*TELECOM_C, EDITED])))
    want = [_key(it) for it in full[:4]]
    root = _assets(want)
    from sors.data.paths import asset_path
    shutil.copytree(asset_path("synth-simple-eval"), root / "synth-simple-eval")
    try:
        sample, _, info = _mod.build_data(_args("synth-v6", eval="simple", datasets_dir=str(root),
                                                passes=f"{full[0].goal}=1"))
    finally:
        shutil.rmtree(root)
    assert info["synth_v5_items"] == 4 and info["synth_v6_selection"]["bindings"] == 4
    assert len(sample(4, random.Random(0))) == 4
    parser = _mod.build_parser()
    assert "-fallback0.5" in _mod.run_tag(parser.parse_args(["--dataset", "synth-v6"]))
