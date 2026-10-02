"""scripts/train.py 的数据组装 (build_data) 测试. 只读数据集, 不加载模型.

跑:  PYTHONPATH=src .venv/bin/python tests/test_train_cli.py
"""

import argparse
import collections
import importlib.util
import json
import pathlib
import random

from _runner import run
from sors.core.menu import row_alignment
from sors.data.jevbench import load_jevbench
from sors.data.paths import asset_path
from sors.data.simple_eval import load_simple_eval
from sors.data.synth import load_synth

_SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "train.py"
_spec = importlib.util.spec_from_file_location("train_cli", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _args(dataset, **kw):
    base = dict(dataset=dataset, k_min=None, k_max=256, k_log=False, k_eval=256, held_out=17, seed=0,
                eval_batch_size=16, eval_limit=0, data_dir="data/banking77", random_codes=0.0, label_smoothing=0.0,
                consistency=0.0, probe_size=200, probe_passes=5, micro_batches=2, eval="banking77+massive+boolq",
                mix=None, mask_descriptions=0.0, passes=None, fallback_rate=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_parse_datasets_accepts_massive():
    assert _mod.parse_datasets("banking77+boolq+massive") == ["banking77", "boolq", "massive"]


def test_v51_dataset_selector_and_legacy_alias_produce_the_same_samples():
    from unittest.mock import patch
    from test_synth_v5_fallback import _marked

    item = _marked()
    with patch("sors.data.synth_v5.load_synth_v5", return_value=[item]):
        current, _, info = _mod.build_data(_args("synth-v5.1", eval="simple", fallback_rate=1.0))
        legacy, _, old_info = _mod.build_data(_args("synth-v5", eval="simple", fallback_rate=1.0))
    assert current(8, random.Random(0)) == legacy(8, random.Random(0))
    assert info == old_info
    args = _mod.build_parser().parse_args(["--dataset", "synth-v5.1", "--fallback-rate", "1"])
    assert "-fallback1" in _mod.run_tag(args)


def test_v51_and_its_legacy_alias_cannot_duplicate_the_dataset():
    try:
        _mod.parse_datasets("synth-v5+synth-v5.1")
    except SystemExit:
        return
    raise AssertionError("the same dataset was accepted twice through aliases")


def test_dataset_defaults_to_synth():
    assert _mod.build_parser().parse_args([]).dataset == "synth"


def test_eval_defaults_to_full_menus_on_banking77_massive_and_boolq():
    """默认评估集与训练集无关: Banking77 test 3080 条配全部 77 类, MASSIVE test 2974 条配全部 60 类,
    BoolQ validation 3270 条. 菜单都是全量、连续编号, 显示原始 label 名."""
    assert _mod.build_parser().parse_args([]).eval == \
        "banking77+banking77-desc+massive+massive-desc+boolq+simple+jevbench"
    _, eval_sets, _ = _mod.build_data(_args("massive"))
    assert set(eval_sets) == {"banking77", "massive", "boolq"}
    b77, mas, bq = eval_sets["banking77"], eval_sets["massive"], eval_sets["boolq"]
    assert len(b77.examples) == 3080 and {len(e.options) for e in b77.examples} == {77}
    assert {e.context_label for e in b77.examples} == {"Customer message"}
    assert "Refund_not_showing_up" in b77.examples[0].option_names
    assert len(mas.examples) == 2974 and {len(e.options) for e in mas.examples} == {60}
    assert {e.context_label for e in mas.examples} == {"Voice command"}
    assert "iot_hue_lightchange" in mas.examples[0].option_names
    assert len(bq.examples) == 3270 and bq.pos_class == 1
    assert all(e.codes is None for es in eval_sets.values() for e in es.examples)


def test_desc_eval_sets_ask_the_same_questions_with_descriptions_on_the_menu():
    """banking77-desc / massive-desc 与 banking77 / massive 逐题相同 (消息、菜单顺序、正确位置),
    只有菜单上显示的文字换成「原始名: description」(description 取自 datasets/label-descriptions)."""
    _, ev, _ = _mod.build_data(_args("massive", eval="banking77+banking77-desc+massive+massive-desc"))
    for raw, desc, f in (("banking77", "banking77-desc", "banking77.json"), ("massive", "massive-desc", "massive.json")):
        d = json.loads(asset_path(f"label-descriptions/{f}").read_text())
        a, b = ev[raw].examples, ev[desc].examples
        assert len(a) == len(b) and ev[desc].pos_class is None
        for x, y in zip(a, b):
            assert (x.query, x.options, x.gold_idx, x.label, x.context_label) == \
                   (y.query, y.options, y.gold_idx, y.label, y.context_label)
            assert y.option_names == [f"{n}: {d[n]}" for n in x.option_names]


def test_eval_takes_a_subset():
    _, eval_sets, _ = _mod.build_data(_args("massive", eval="massive"))
    assert set(eval_sets) == {"massive"}


def test_eval_simple_adds_one_set_per_menu_size_straight_from_the_file():
    """simple 展开成 simple5 ... simple255 与 simple_bool, 题目与 load_simple_eval 逐题相同; 二元那档报 AUROC."""
    _, eval_sets, _ = _mod.build_data(_args("massive", eval="simple"))
    want = load_simple_eval()
    assert list(eval_sets) == list(want)
    assert all(eval_sets[k].examples == want[k] for k in want)
    assert eval_sets["simple_bool"].pos_class == 1 and eval_sets["simple255"].pos_class is None


def test_eval_jevbench_adds_one_set_per_public_tier_straight_from_the_files():
    """jevbench 展开成 jevbench_easy / _original / _hard, 题目与 load_jevbench 逐题相同. 三档都混着 noul / choice / score,
    不报二元那组. hard 的提示最长 3838 token, 批只取四分之一, 一批的 token 数与 BoolQ 那档相当."""
    _, eval_sets, _ = _mod.build_data(_args("massive", eval="jevbench"))
    want = load_jevbench()
    assert list(eval_sets) == list(want) == ["jevbench_easy", "jevbench_original", "jevbench_hard"]
    assert all(eval_sets[k].examples == want[k] and eval_sets[k].pos_class is None for k in want)
    assert [eval_sets[k].batch_size for k in want] == [16, 16, 4]


def test_eval_rejects_an_unknown_set():
    try:
        _mod.build_data(_args("massive", eval="massive+synth"))
    except SystemExit:
        return
    raise AssertionError("--eval massive+synth accepted; synth is training data")


def test_synth_trains_domain_menus_and_binary_questions_half_and_half():
    """--dataset synth: 一批 8 条 = 4 道菜单题 (正确意图所在领域的 256 项全量菜单) + 4 道二元题 (no / yes, 问消息里的细节)."""
    sample_fn, _, info = _mod.build_data(_args("synth", eval="massive"))
    _, domains = load_synth()
    assert info == {"synth_classes": 4096}
    rng = random.Random(0)
    for _ in range(50):
        batch = sample_fn(8, rng)
        choice = [e for e in batch if e.qtype == "choice"]
        binary = [e for e in batch if e.qtype == "bool"]
        assert len(choice) == len(binary) == 4
        assert all(len(e.options) == 256 and {domains[c] for c in e.options} == {domains[e.label]} for e in choice)
        assert all(sorted(e.option_names) == ["no", "yes"] and e.question.endswith("?") for e in binary)
        assert {e.context_label for e in batch} == {"Customer message"}


def test_synth_menu_trains_the_domain_menus_without_the_binary_questions():
    """--dataset synth-menu: synth 去掉二元题, 一批 8 条全是菜单题 (正确意图所在领域的 256 项全量菜单).
    二元题的答案全落在 D0 / D1, 去掉它们才看得出这一半批次是否把概率拉向前两格."""
    sample_fn, _, info = _mod.build_data(_args("synth-menu", eval="massive"))
    _, domains = load_synth()
    assert info == {"synth_classes": 4096}
    rng = random.Random(0)
    for _ in range(50):
        batch = sample_fn(8, rng)
        assert len(batch) == 8 and {e.qtype for e in batch} == {"choice"}
        assert all(len(e.options) == 256 and {domains[c] for c in e.options} == {domains[e.label]} for e in batch)


def test_synth_and_synth_menu_cannot_be_combined():
    """两者的菜单题是同一批, 并用会让它们在一批里占两份."""
    try:
        _mod.parse_datasets("synth+synth-menu")
    except SystemExit:
        return
    raise AssertionError("--dataset synth+synth-menu accepted")


def test_banking77_and_synth_keep_separate_menus():
    """两个都在训练里时不再并池: Banking77 的题只列它的 60 个训练类, 合成意图的题只列自己领域的 256 个."""
    sample_fn, eval_sets, _ = _mod.build_data(_args("banking77+synth", eval="massive"))
    rng = random.Random(0)
    sizes = collections.Counter(len(e.options) for _ in range(50) for e in sample_fn(8, rng) if e.qtype == "choice")
    assert set(sizes) == {60, 256}, sizes
    assert set(eval_sets) == {"seen", "unseen", "massive"}


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


def test_random_codes_lets_60_item_menus_train_every_code_and_leaves_eval_contiguous():
    """--random-codes 0.5 配 train3 的数据 (菜单 60 项 / BoolQ 2 项): 约一半的题换成随机码, 二元题也一样,
    正确答案的码铺满 D0..D255. 评估集照旧按位置编号, 与部署时的菜单同形."""
    sample_fn, eval_sets, _ = _mod.build_data(_args("banking77+boolq+massive", random_codes=0.5))
    rng = random.Random(0)
    exs = [e for _ in range(500) for e in sample_fn(8, rng)]
    for qtype in ("choice", "bool"):
        got = [e for e in exs if e.qtype == qtype]
        share = sum(e.codes is not None for e in got) / len(got)
        assert abs(share - 0.5) < 0.05, (qtype, share)
    assert {e.slot_codes[e.gold_idx] for e in exs if e.qtype == "choice"} == set(range(256))
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


def test_synth_v3_trains_its_five_domains_alongside_synth():
    """--dataset synth+synth-v3: 一批 9 条 = synth 菜单题 3 + synth 二元题 3 + v3 3 (第一个 sampler 拿零头).
    v3 那几条的上下文标签是各领域的 LABEL, 菜单就是题自己的选项 (2..107 项), 不与 synth 的意图混."""
    sample_fn, _, info = _mod.build_data(_args("synth+synth-v3", eval="massive"))
    assert info == {"synth_classes": 4096, "synth_v3_items": 688}
    labels = {"Support ticket", "Hotel document", "Browser agent state", "Security on-call screen", "Code and CI state"}
    rng = random.Random(0)
    seen = set()
    for _ in range(100):
        batch = sample_fn(9, rng)
        v3 = [e for e in batch if e.context_label in labels]
        assert len(v3) == 3, [e.context_label for e in batch]
        assert all(2 <= len(e.options) <= 107 for e in v3)
        seen |= {e.context_label for e in v3}
    assert seen == labels



V5_LABELS = {"Customer message", "Support ticket", "Hotel document", "Browser agent state", "Security on-call screen",
             "Code and CI state", ""}


def test_synth_v5_draws_every_goal_and_shows_intent_menus_as_key_and_description():
    """v5.1 的六个目标默认等概率；每次抽取一个菜单版本，256行题的份额约为1/6。
    intent 是「键: 描述」，规则材料与自然消息的上下文标题都会出现。"""
    sample_fn, _, info = _mod.build_data(_args("synth-v5", eval="massive"))
    assert info["synth_v5_items"] == 67939, info
    assert info["synth_v5_mix"] == [(0, {"ambiguous": 1.0, "breadth": 1.0, "complex": 1.0, "edge_case": 1.0,
                                         "long_context": 1.0, "long_menu": 1.0})]
    full_menus = {}
    for path in asset_path("synth-intents-v5.1").glob("*.questions.json"):
        for question in json.loads(path.read_text())["questions"]:
            if len(question["options"]) == 256:
                rows = frozenset(f"{key}: {description}" for key, description in question["options"].items())
                full_menus[rows] = set(question["ask"])
    rng = random.Random(0)
    labels, sizes = set(), collections.Counter()
    for _ in range(60):
        for e in sample_fn(8, rng):
            labels.add(e.context_label)
            sizes["256" if len(e.options) == 256 else "other"] += 1
            if len(e.options) == 256:
                assert e.question in full_menus[frozenset(e.option_names)], e.question
    assert labels <= V5_LABELS and "" in labels and "Customer message" in labels, labels
    assert 0.10 < sizes["256"] / sum(sizes.values()) < 0.20, sizes


def test_synth_v5_mix_sets_the_goal_shares():
    sample_fn, _, info = _mod.build_data(_args("synth-v5", eval="massive",
                                               mix="long_menu=14,breadth=2,complex=2,edge_case=1,long_context=0.9,"
                                                   "ambiguous=0.1"))
    rng = random.Random(0)
    n = long = 0
    # 10k 次抽取: 占比的标准差约 0.0046, ±0.05 的范围有 10 倍余量. 只抽 1000 次时标准差 0.0145,
    # 数据一变随机流就变, 约 1/300 的种子会落到范围外 (government 合并时 seed 0 抽到 0.752)
    for _ in range(1000):
        for e in sample_fn(10, rng):
            n += 1
            long += len(e.options) == 256
    assert 0.65 < long / n < 0.75, long / n


def test_a_mix_that_drops_a_goal_or_comes_without_synth_v5_is_refused():
    for ds, mix in (("synth-v5", "long_menu=1,breadth=1"), ("synth", "long_menu=1")):
        try:
            _mod.build_data(_args(ds, eval="massive", mix=mix))
        except SystemExit:
            continue
        raise AssertionError(f"--dataset {ds} --mix {mix!r} accepted")


def test_mask_descriptions_must_be_a_share_and_needs_synth_v5():
    for ds, rate in (("synth-v5", 1.5), ("synth-v5", -0.1), ("synth", 0.5)):
        try:
            _mod.build_data(_args(ds, eval="massive", mask_descriptions=rate))
        except SystemExit:
            continue
        raise AssertionError(f"--dataset {ds} --mask-descriptions {rate} accepted")


def test_mask_descriptions_hides_intent_descriptions_on_both_copies_of_a_pair():
    """--mask-descriptions 1 --consistency 1: intent 题 (maskable) 的两份都只剩键, 按描述对齐照样成立."""
    sample_fn, _, _ = _mod.build_data(_args("synth-v5", eval="massive", mask_descriptions=1.0, consistency=1.0,
                                            mix="long_menu=96,breadth=1,complex=1,edge_case=1,long_context=1,ambiguous=1"))
    rng = random.Random(0)
    pairs = 0
    for _ in range(10):
        batch = sample_fn(8, rng)
        for a, b in zip(batch[0::2], batch[1::2]):
            if len(a.options) == 256:
                pairs += 1
                assert not any(": " in n for n in a.option_names + b.option_names)
                assert [b.options[j] for j in row_alignment(a, b)] == a.options
    assert pairs > 60, pairs


def test_consistency_pairs_a_v5_material_with_another_phrasing_of_it():
    """v4 的材料带两种说法: --consistency 下配对的两份读的是不同说法, 问句与选项集合相同."""
    sample_fn, _, _ = _mod.build_data(_args("synth-v5", eval="massive", consistency=1.0,
                                            mix="long_menu=1,breadth=1,complex=33,edge_case=33,long_context=31,"
                                                "ambiguous=1"))
    rng = random.Random(0)
    reworded = 0
    for _ in range(20):
        batch = sample_fn(8, rng)
        for a, b in zip(batch[0::2], batch[1::2]):
            assert a.question == b.question and sorted(a.options) == sorted(b.options)
            assert [b.options[j] for j in row_alignment(a, b)] == a.options
            reworded += a.query != b.query
    assert reworded > 120, reworded


def test_synth_v5_random_other_view_keeps_one_example_per_draw():
    sample_fn, _, _ = _mod.build_data(_args("synth-v5", eval="simple",
                                            mix="long_menu=1,breadth=1,complex=1,edge_case=1,long_context=1,ambiguous=95"))
    rng = random.Random(0)
    assert all(len(sample_fn(8, rng)) == 8 for _ in range(40))


def test_fallback_cli_reaches_both_samplers_and_keeps_consistency_views_equal():
    from unittest.mock import patch
    from test_synth_v5_fallback import _marked

    item = _marked()
    for passes in (None, "breadth=1"):
        for rate, size in ((0.0, 2), (1.0, 3)):
            args = _mod.build_parser().parse_args(["--dataset", "synth-v5", "--eval", "simple",
                                                   "--fallback-rate", str(rate), "--consistency", "1"])
            args.passes = passes
            with patch("sors.data.synth_v5.load_synth_v5", return_value=[item]):
                sample, _, info = _mod.build_data(args)
            assert info["synth_v5_fallback_rate"] == rate
            batch = sample(8, random.Random(0))
            assert len(batch) == 16 and all(len(ex.options) == size for ex in batch)
            for a, b in zip(batch[::2], batch[1::2]):
                assert a.option_names[a.gold_idx] == b.option_names[b.gold_idx]
                assert [b.options[j] for j in row_alignment(a, b)] == a.options


def test_fallback_cli_rejects_bad_rates_and_requires_v5():
    for dataset, rate in (("synth", 0.5), ("synth-v5", -0.1), ("synth-v5", 1.1), ("synth-v5", float("nan"))):
        try:
            _mod.build_data(_args(dataset, fallback_rate=rate, eval="simple"))
        except SystemExit as exc:
            assert "fallback" in str(exc)
            continue
        raise AssertionError(f"accepted {dataset=} {rate=}")


def test_fallback_rate_is_recorded_in_the_run_tag_for_v5():
    parser = _mod.build_parser()
    assert "-fallback0.5" in _mod.run_tag(parser.parse_args(["--dataset", "synth-v5"]))
    assert "-fallback0" in _mod.run_tag(parser.parse_args(["--dataset", "synth-v5", "--fallback-rate", "0"]))


def test_synth_v5_options_are_named_in_the_run_directory():
    p = _mod.build_parser()
    args = p.parse_args(["--dataset", "synth-v5", "--mask-descriptions", "0.5", "--mix", "long_menu=4,breadth=1"])
    vars(args).update(_mod.resolve_adapter(args))
    tag = _mod.run_tag(args)
    assert "-mask0.5" in tag and "-mix" in tag, tag
    other = p.parse_args(["--dataset", "synth-v5", "--mask-descriptions", "0.5", "--mix", "long_menu=3,breadth=1"])
    vars(other).update(_mod.resolve_adapter(other))
    assert _mod.run_tag(other) != tag


def test_passes_draw_synth_v5_in_rounds_and_the_hard_goals_repeat():
    """v5.1 有67939个绑定、742对菜单版本，一轮67197次抽取；
    complex / edge_case / long_context 共4043个绑定各多出两遍。"""
    _, _, info = _mod.build_data(_args("synth-v5", eval="massive", passes="complex=3,edge_case=3,long_context=3"))
    assert info["synth_v5_round"] == 67197 + 2 * 4043, info
    assert info["synth_v5_passes"] == {"ambiguous": 1, "breadth": 1, "complex": 3, "edge_case": 3,
                                       "long_context": 3, "long_menu": 1}
    assert "synth_v5_mix" not in info


def test_passes_are_refused_with_a_mix_without_synth_v5_or_when_unreadable():
    for ds, kw in (("synth-v5", dict(passes="complex=3", mix="long_menu=1")), ("synth", dict(passes="complex=3")),
                   ("synth-v5", dict(passes="complex=1.5"))):
        try:
            _mod.build_data(_args(ds, eval="massive", **kw))
        except SystemExit:
            continue
        raise AssertionError(f"--dataset {ds} {kw} accepted")


def test_passes_are_named_in_the_run_directory():
    p = _mod.build_parser()
    a = p.parse_args(["--dataset", "synth-v5", "--passes", "complex=3"])
    b = p.parse_args(["--dataset", "synth-v5", "--passes", "complex=4"])
    assert "-pass" in _mod.run_tag(a) and _mod.run_tag(a) != _mod.run_tag(b)
    assert "-pass" not in _mod.run_tag(p.parse_args(["--dataset", "synth-v5"]))


def test_max_length_defaults_to_8192():
    assert _mod.build_parser().parse_args([]).max_length == 8192


def test_label_smoothing_defaults_to_zero_and_is_named_in_the_run_directory():
    p = _mod.build_parser()
    assert p.parse_args([]).label_smoothing == 0.0
    args = p.parse_args(["--label-smoothing", "0.1"])
    vars(args).update(_mod.resolve_adapter(args))
    assert _mod.run_tag(args) == "-kfull256-ls0.1-vocab"


def test_label_smoothing_outside_zero_to_one_is_refused():
    try:
        _mod.build_data(_args("boolq", label_smoothing=1.0))
    except SystemExit:
        return
    raise AssertionError("--label-smoothing 1.0 accepted")


def test_random_codes_rejects_a_rate_outside_zero_to_one():
    try:
        _mod.build_data(_args("boolq", random_codes=1.5))
    except SystemExit:
        return
    raise AssertionError("--random-codes 1.5 accepted")


def test_consistency_defaults_to_zero_and_is_named_in_the_run_directory():
    p = _mod.build_parser()
    assert p.parse_args([]).consistency == 0.0
    args = p.parse_args(["--consistency", "1", "--label-smoothing", "0.1"])
    vars(args).update(_mod.resolve_adapter(args))
    assert _mod.run_tag(args) == "-kfull256-ls0.1-js1-vocab"


def test_consistency_below_zero_is_refused():
    try:
        _mod.build_data(_args("boolq", consistency=-0.5))
    except SystemExit:
        return
    raise AssertionError("--consistency -0.5 accepted")


def test_consistency_gives_every_question_a_partner_in_another_row_order():
    """--consistency > 0: sample_fn(9) 给 9 道题各两份, 共 18 条, 同一道题的两份挨着 (2i, 2i+1).
    菜单题、二元题、v3 都配对; 两份上下文、问句、选项集合相同, 行序不同. --random-codes 1 下两份各换各的码,
    二元题也一样."""
    sample_fn, _, _ = _mod.build_data(_args("synth+synth-v3", consistency=1.0, random_codes=1.0, eval="massive"))
    rng = random.Random(0)
    kinds = collections.Counter()
    for _ in range(30):
        batch = sample_fn(9, rng)
        assert len(batch) == 18
        for a, b in zip(batch[0::2], batch[1::2]):
            assert (a.query, a.question, a.context_label, a.qtype) == (b.query, b.question, b.context_label, b.qtype)
            assert sorted(a.options) == sorted(b.options) and a.options != b.options
            assert [b.options[j] for j in row_alignment(a, b)] == a.options
            assert a.codes is not None and b.codes is not None
            kinds["v3" if a.context_label != "Customer message" else len(a.options)] += 1
    assert kinds[256] == kinds[2] == kinds["v3"] == 90, kinds


def test_probe_defaults_to_200_questions_in_5_arrangements():
    args = _mod.build_parser().parse_args([])
    assert (args.probe_size, args.probe_passes) == (200, 5)


def test_probe_needs_a_question_and_two_arrangements():
    """一份排法没有东西可比: agree 恒为 1、js 恒为 0."""
    for kw in ({"probe_passes": 1}, {"probe_size": 0}):
        try:
            _mod.build_data(_args("boolq", **kw))
        except SystemExit:
            continue
        raise AssertionError(f"{kw} accepted")


def test_loss_defaults_to_the_whole_vocabulary():
    assert _mod.build_parser().parse_args([]).loss == "vocab"


def test_micro_batches_default_to_two_and_stay_out_of_the_run_directory():
    """分组前向只改算法不改数学 (每行的 logits 与整批一次相同), 所以默认就开, 目录名不记它."""
    p = _mod.build_parser()
    assert p.parse_args([]).micro_batches == 2
    assert _mod.run_tag(p.parse_args(["--micro-batches", "4"])) == _mod.run_tag(p.parse_args([]))


def test_micro_batches_below_one_are_refused():
    try:
        _mod.build_data(_args("boolq", micro_batches=0))
    except SystemExit:
        return
    raise AssertionError("--micro-batches 0 accepted")


def test_run_tag_names_the_loss_and_keeps_the_old_names_for_old_losses():
    """vocab 加 -vocab; all-slots 仍是 -allslots, menu 仍不加 —— 旧 run 的目录名照旧能复现."""
    p = _mod.build_parser()
    tag = lambda *argv: _mod.run_tag(p.parse_args(list(argv)))  # noqa: E731
    assert tag() == "-kfull256-vocab"
    assert tag("--loss", "all-slots", "--random-codes", "0.8") == "-kfull256-rcodes0.8-allslots"
    assert tag("--loss", "menu", "--k-min", "2", "--k-max", "10") == "-k2-10"


def test_context_marker_defaults_off_and_is_named_in_the_run_directory_after_the_type_marker():
    p = _mod.build_parser()
    assert p.parse_args([]).context_marker is False
    assert _mod.run_tag(p.parse_args(["--context-marker"])) == "-ctx-kfull256-vocab"
    assert _mod.run_tag(p.parse_args(["--type-marker", "--context-marker"])) == "-qtype-ctx-kfull256-vocab"


def _checkpoint(trainable, r, alpha):
    """用一层的随机 Qwen3 走真的 prepare_model / save_trained 存一份档."""
    import tempfile

    from transformers import Qwen3Config, Qwen3ForCausalLM

    from sors.core.checkpoint import save_trained
    from sors.core.model import prepare_model
    from sors.training.loop import TrainConfig

    cfg = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=8)
    ids = list(range(40, 64))
    m = prepare_model(Qwen3ForCausalLM(cfg), ids, lora_r=r, lora_alpha=alpha, lora_dropout=0.0, trainable=trainable)
    f = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
    save_trained(m, ids, TrainConfig(), f.name)
    return f.name


def test_adapter_without_init_defaults_to_attention_r8_alpha16():
    p = _mod.build_parser()
    assert _mod.resolve_adapter(p.parse_args([])) == {"trainable": "attn", "lora_r": 8, "lora_alpha": 16}
    assert _mod.resolve_adapter(p.parse_args(["--lora-r", "32", "--lora-alpha", "64", "--trainable", "attn-mlp"])) \
        == {"trainable": "attn-mlp", "lora_r": 32, "lora_alpha": 64}
    # d-only 没有 LoRA, 与存档里记的一样写 None, result.json 不报一个没用上的 rank
    assert _mod.resolve_adapter(p.parse_args(["--trainable", "d-only"])) \
        == {"trainable": "d-only", "lora_r": None, "lora_alpha": None}


def test_adapter_with_init_comes_from_the_checkpoint():
    """--init 只评估或接着训时, 不必再在命令行上复述那次训练的 LoRA 形状."""
    path = _checkpoint("attn-mlp", 4, 12)
    got = _mod.resolve_adapter(_mod.build_parser().parse_args(["--init", path]))
    assert got == {"trainable": "attn-mlp", "lora_r": 4, "lora_alpha": 12}


def test_adapter_with_init_rejects_a_flag_that_contradicts_the_checkpoint():
    path = _checkpoint("attn", 4, 12)
    try:
        _mod.resolve_adapter(_mod.build_parser().parse_args(["--init", path, "--lora-alpha", "16"]))
    except SystemExit:
        return
    raise AssertionError("--lora-alpha 16 accepted for a checkpoint trained with alpha 12")


def test_run_tag_names_rank_and_alpha_only_when_they_leave_8_and_16():
    p = _mod.build_parser()

    def tag(*argv):
        args = p.parse_args(list(argv))
        vars(args).update(_mod.resolve_adapter(args))
        return _mod.run_tag(args)

    assert tag() == "-kfull256-vocab"
    assert tag("--lora-r", "32", "--lora-alpha", "64") == "-r32-alpha64-kfull256-vocab"
    assert tag("--lora-r", "32") == "-r32-kfull256-vocab"
    assert tag("--trainable", "d-only") == "-kfull256-vocab"


def test_trainable_full_has_no_rank_and_names_its_learning_rate():
    """full 没有 LoRA, rank 与 alpha 同 d-only 记 None. 主干学习率与 LoRA 共用 --lr-lora,
    离开默认的 1e-4 时写进目录名 (全参要小一个量级, 不写的话目录名看不出来)."""
    p = _mod.build_parser()
    args = p.parse_args(["--trainable", "full", "--lr-lora", "1e-5"])
    vars(args).update(_mod.resolve_adapter(args))
    assert (args.trainable, args.lora_r, args.lora_alpha) == ("full", None, None)
    assert _mod.run_tag(args) == "-lr1e-05-kfull256-vocab"
    assert _mod.run_tag(p.parse_args(["--lr-lora", "1e-4"])) == "-kfull256-vocab"


def test_save_every_defaults_to_every_eval_point():
    """不给 --save-every 时每个评估点存一次, 与 --eval-every 同步; 给了就照给的步数, 0 = 途中不存."""
    p = _mod.build_parser()
    assert _mod.resolve_save_every(p.parse_args(["--eval-every", "250"])) == 250
    assert _mod.resolve_save_every(p.parse_args(["--eval-every", "250", "--save-every", "100"])) == 100
    assert _mod.resolve_save_every(p.parse_args(["--save-every", "0"])) == 0


def test_save_every_rejects_a_negative_step_count():
    try:
        _mod.resolve_save_every(_mod.build_parser().parse_args(["--save-every", "-1"]))
    except SystemExit:
        return
    raise AssertionError("--save-every -1 accepted")


def test_checkpoint_path_sorts_by_step():
    """途中的档放在 checkpoints/ 下, 步数补零, ls 出来就是训练顺序."""
    out = pathlib.Path("runs/x")
    assert _mod.checkpoint_path(out, 250) == out / "checkpoints" / "step-00250.safetensors"
    assert sorted([_mod.checkpoint_path(out, s) for s in (1000, 250, 500)]) \
        == [_mod.checkpoint_path(out, s) for s in (250, 500, 1000)]


if __name__ == "__main__":
    run(globals())
