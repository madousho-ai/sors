"""scripts/build-synth-v6.py: 由 synth-intents-v6/spec.json 与 drop.json 写 selection.json; --check 重算并比对."""
import importlib.util
import json
import pathlib
import shutil
import tempfile

import pytest

from sors.data.synth_v6 import load_synth_v6
from test_synth_v5 import EDITED, TELECOM_C, TELECOM_Q, _data

_SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "build-synth-v6.py"
_spec = importlib.util.spec_from_file_location("build_synth_v6", _SCRIPT)
build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build)


def _root():
    root = pathlib.Path(tempfile.mkdtemp())
    src = _data(telecom=(TELECOM_Q, [*TELECOM_C, EDITED]))
    shutil.copytree(src, root / "synth-intents-v5.25")
    shutil.rmtree(src)
    (root / "held").mkdir()
    (root / "held" / "heldout.json").write_text(json.dumps({"contexts": ["telecom_queue_a"]}))
    v6 = root / "synth-intents-v6"
    v6.mkdir()
    (v6 / "spec.json").write_text(json.dumps({"source": "synth-intents-v5.25", "classes": {"base": ["base"]},
                                              "n": {"base": 100}, "eval_n": 1, "salt": "t:"}))
    (v6 / "drop.json").write_text(json.dumps({"contexts": ["telecom_rule_1"], "also": ["held/heldout.json"]}))
    return root


def test_build_writes_a_selection_the_loader_reads_without_the_dropped_materials_or_their_edits():
    root = _root()
    try:
        assert build.main(["--datasets-dir", str(root)]) == 0
        sel = json.loads((root / "synth-intents-v6" / "selection.json").read_text())
        assert sel["source"] == "synth-intents-v5.25" and sel["spec"]["n"] == {"base": 100}
        items = load_synth_v6(datasets_dir=root)
        ids = {it.context_id for it in items}
        assert len(items) == len(sel["bindings"]) > 0
        assert not ids & {"telecom_rule_1", "telecom_queue_a", "telecom_queue_a/cf"}
    finally:
        shutil.rmtree(root)


def test_check_passes_on_a_fresh_build_and_fails_after_the_spec_changes():
    root = _root()
    try:
        build.main(["--datasets-dir", str(root)])
        assert build.main(["--datasets-dir", str(root), "--check"]) == 0
        spec = root / "synth-intents-v6" / "spec.json"
        s = json.loads(spec.read_text())
        s["eval_n"] = 2
        spec.write_text(json.dumps(s))
        assert build.main(["--datasets-dir", str(root), "--check"]) == 1
    finally:
        shutil.rmtree(root)


def test_a_drop_reference_that_does_not_exist_is_an_error():
    root = _root()
    try:
        (root / "held" / "heldout.json").unlink()
        with pytest.raises(FileNotFoundError):
            build.main(["--datasets-dir", str(root)])
    finally:
        shutil.rmtree(root)
