"""scripts/train.py 的数据组装 (build_data) 测试. 只读数据集, 不加载模型.

跑:  PYTHONPATH=src .venv/bin/python tests/test_train_cli.py
"""

import argparse
import collections
import importlib.util
import pathlib
import random

from _runner import run

_SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "train.py"
_spec = importlib.util.spec_from_file_location("train_cli", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _args(dataset, **kw):
    base = dict(dataset=dataset, k_min=None, k_max=256, k_log=False, k_eval=256, held_out=17, seed=0,
                eval_batch_size=16, eval_limit=0, data_dir="data/banking77")
    base.update(kw)
    return argparse.Namespace(**base)


def test_parse_datasets_accepts_massive():
    assert _mod.parse_datasets("banking77+boolq+massive") == ["banking77", "boolq", "massive"]


def test_banking77_boolq_massive_trains_only_on_slots_d0_to_d59():
    """两个意图集的上下文标签不同, 不共用选项池: 各自 60 项全量菜单, 正确答案只落在 D0..D59.
    一批 8 条按 banking77 4 / massive 2 / boolq 2 分, 与 run2 的配比相同."""
    sample_fn, _, _ = _mod.build_data(_args("banking77+boolq+massive"))
    rng = random.Random(0)
    gold_max, sizes, per_batch = 0, collections.Counter(), collections.Counter()
    for _ in range(300):
        batch = sample_fn(8, rng)
        per_batch[tuple(sorted(collections.Counter(e.context_label for e in batch).items()))] += 1
        for e in batch:
            sizes[(e.context_label, len(e.options))] += 1
            gold_max = max(gold_max, e.gold_idx)
    assert gold_max == 59, gold_max
    assert set(sizes) == {("Customer message", 60), ("Voice command", 60), ("Passage", 2)}, sizes
    assert set(per_batch) == {(("Customer message", 4), ("Passage", 2), ("Voice command", 2))}, per_batch


if __name__ == "__main__":
    run(globals())
