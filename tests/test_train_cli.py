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
                eval_batch_size=16, eval_limit=0, eval_synth=False, data_dir="data/banking77", random_codes=0.0)
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


def test_eval_synth_adds_full_60_and_256_menus_over_all_512_synth_intents():
    """--eval-synth: 1024 条合成意图消息各配一个菜单, 选项全来自 512 个合成意图.
    synth60 的正确答案都在训练覆盖的 D0..D59 里, synth256 的散到 D0..D255."""
    _, eval_sets, _ = _mod.build_data(_args("banking77+boolq+massive", eval_synth=True))
    s60, s256 = eval_sets["synth60"], eval_sets["synth256"]
    assert len(s60.examples) == len(s256.examples) == 1024
    assert {len(e.options) for e in s60.examples} == {60}
    assert {len(e.options) for e in s256.examples} == {256}
    assert max(e.gold_idx for e in s256.examples) >= 250
    assert sum(e.gold_idx >= 60 for e in s256.examples) > 700, "约 196/256 的题正确答案落在训练没覆盖的码上"


def test_eval_synth_refuses_when_synth_is_in_training():
    try:
        _mod.build_data(_args("banking77+synth", eval_synth=True))
    except SystemExit:
        return
    raise AssertionError("synth 在训练里时 --eval-synth 不是留出评估, 应当拒绝")


def test_random_codes_lets_60_item_menus_train_every_code_and_leaves_eval_contiguous():
    """--random-codes 0.5 配 train3 的数据 (菜单 60 项 / BoolQ 2 项): 约一半选择题换成随机码,
    正确答案的码铺满 D0..D255; BoolQ 永远 D0 / D1. 评估集照旧按位置编号, 与部署时的菜单同形."""
    sample_fn, eval_sets, _ = _mod.build_data(_args("banking77+boolq+massive", random_codes=0.5))
    rng = random.Random(0)
    exs = [e for _ in range(500) for e in sample_fn(8, rng)]
    choice = [e for e in exs if e.qtype == "choice"]
    share = sum(e.codes is not None for e in choice) / len(choice)
    assert abs(share - 0.5) < 0.05, share
    assert all(e.codes is None for e in exs if e.qtype == "bool")
    assert {e.slot_codes[e.gold_idx] for e in choice} == set(range(256))
    assert all(e.codes is None for es in eval_sets.values() for e in es.examples)


def test_random_codes_over_the_whole_run_answer_evenly_without_favouring_any_code():
    """--random-codes 0.8 跑满 3000 步 (18000 道选择题, 两成连续编号):
    换码菜单上每个码出现时是答案的概率都是 1/60, D0..D59 与 D60..D255 一样 (旧的均衡法差 5 倍);
    整场下来两段每个码当答案的平均次数差不到 6% (均匀挑码时差 1.8 倍). 计数要跨 batch 保留, 每批重新计数就补不齐."""
    sample_fn, _, _ = _mod.build_data(_args("banking77+boolq+massive", random_codes=0.8))
    rng = random.Random(0)
    choice = [e for _ in range(3000) for e in sample_fn(8, rng) if e.qtype == "choice"]
    rand = [e for e in choice if e.codes is not None]
    for lo, hi in ((0, 60), (60, 256)):
        on = sum(lo <= c < hi for e in rand for c in e.slot_codes)
        gold = sum(lo <= e.slot_codes[e.gold_idx] < hi for e in rand)
        assert abs(gold / on * 60 - 1) < 0.15, (lo, hi, gold, on)
    counts = [0] * 256
    for e in choice:
        counts[e.slot_codes[e.gold_idx]] += 1
    lo, hi = sum(counts[:60]) / 60, sum(counts[60:]) / 196
    assert abs(lo / hi - 1) < 0.06, (lo, hi)


def test_random_codes_rejects_a_rate_outside_zero_to_one():
    try:
        _mod.build_data(_args("boolq", random_codes=1.5))
    except SystemExit:
        return
    raise AssertionError("--random-codes 1.5 accepted")


if __name__ == "__main__":
    run(globals())
