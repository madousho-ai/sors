"""--no-initial-eval: 跳过开训前 (step 0) 那次评估. 直接 python tests/test_initial_eval.py."""

import importlib.util
import pathlib
from dataclasses import asdict, replace

from _runner import run
from test_evaluate import _tiny
from test_resume import _Sampler, _training_config
from sors.training.loop import TrainConfig, train

_SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "train.py"
_spec = importlib.util.spec_from_file_location("train_cli_initial_eval", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_the_initial_eval_runs_by_default():
    assert TrainConfig().initial_eval is True
    assert _mod.build_parser().parse_args([]).initial_eval is True
    tok, ids, m = _tiny()
    hist = train(m, tok, ids, _Sampler(), {}, _training_config())
    assert [h["step"] for h in hist] == [0, 2, 4]


def test_skipping_the_initial_eval_drops_only_the_step_0_record():
    assert _mod.build_parser().parse_args(["--no-initial-eval"]).initial_eval is False
    tok, ids, m = _tiny()
    hist = train(m, tok, ids, _Sampler(), {}, replace(_training_config(), initial_eval=False))
    assert [h["step"] for h in hist] == [2, 4]


def test_an_eval_only_run_still_evaluates_when_the_initial_eval_is_skipped():
    tok, ids, m = _tiny()
    hist = train(m, tok, ids, _Sampler(), {}, replace(_training_config(), steps=0, initial_eval=False))
    assert [h["step"] for h in hist] == [0]


def test_the_initial_eval_switch_may_change_on_resume():
    """只决定开头评不评, 不影响训练本身, 续跑时改它不该被当成配置变了."""
    cfg = _training_config()
    saved = {"step": 2, "config": asdict(cfg)}
    tok, ids, m = _tiny()
    train(m, tok, ids, _Sampler(), {}, replace(cfg, initial_eval=False), resume=saved)


if __name__ == "__main__":
    run(globals())
