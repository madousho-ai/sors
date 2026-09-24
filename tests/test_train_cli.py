"""scripts/train.py 的数据组装 (build_data) 测试. 只读数据集, 不加载模型.

跑:  PYTHONPATH=src .venv/bin/python tests/test_train_cli.py
"""

import argparse
import collections
import importlib.util
import pathlib
import random

from _runner import run
from decidophobia.simple_eval import load_simple_eval
from decidophobia.synth import load_synth

_SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "train.py"
_spec = importlib.util.spec_from_file_location("train_cli", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _args(dataset, **kw):
    base = dict(dataset=dataset, k_min=None, k_max=256, k_log=False, k_eval=256, held_out=17, seed=0,
                eval_batch_size=16, eval_limit=0, data_dir="data/banking77", random_codes=0.0,
                eval="banking77+massive+boolq")
    base.update(kw)
    return argparse.Namespace(**base)


def test_parse_datasets_accepts_massive():
    assert _mod.parse_datasets("banking77+boolq+massive") == ["banking77", "boolq", "massive"]


def test_dataset_defaults_to_synth():
    assert _mod.build_parser().parse_args([]).dataset == "synth"


def test_eval_defaults_to_full_menus_on_banking77_massive_and_boolq():
    """默认评估集与训练集无关: Banking77 test 3080 条配全部 77 类, MASSIVE test 2974 条配全部 60 类,
    BoolQ validation 3270 条. 菜单都是全量、连续编号."""
    assert _mod.build_parser().parse_args([]).eval == "banking77+massive+boolq+simple"
    _, eval_sets, _ = _mod.build_data(_args("massive"))
    assert set(eval_sets) == {"banking77", "massive", "boolq"}
    b77, mas, bq = eval_sets["banking77"], eval_sets["massive"], eval_sets["boolq"]
    assert len(b77.examples) == 3080 and {len(e.options) for e in b77.examples} == {77}
    assert {e.context_label for e in b77.examples} == {"Customer message"}
    assert len(mas.examples) == 2974 and {len(e.options) for e in mas.examples} == {60}
    assert {e.context_label for e in mas.examples} == {"Voice command"}
    assert len(bq.examples) == 3270 and bq.pos_class == 1
    assert all(e.codes is None for es in eval_sets.values() for e in es.examples)


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


def test_loss_defaults_to_the_whole_vocabulary():
    assert _mod.build_parser().parse_args([]).loss == "vocab"


def test_run_tag_names_the_loss_and_keeps_the_old_names_for_old_losses():
    """vocab 加 -vocab; all-slots 仍是 -allslots, menu 仍不加 —— 旧 run 的目录名照旧能复现."""
    p = _mod.build_parser()
    tag = lambda *argv: _mod.run_tag(p.parse_args(list(argv)))  # noqa: E731
    assert tag() == "-kfull256-vocab"
    assert tag("--loss", "all-slots", "--random-codes", "0.8") == "-kfull256-rcodes0.8-allslots"
    assert tag("--loss", "menu", "--k-min", "2", "--k-max", "10") == "-k2-10"


def _checkpoint(trainable, r, alpha):
    """用一层的随机 Qwen3 走真的 prepare_model / save_trained 存一份 trained.pt."""
    import tempfile

    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from decidophobia.model import prepare_model
    from decidophobia.train import TrainConfig, save_trained

    cfg = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=8)
    ids = list(range(40, 64))
    m = prepare_model(Qwen3ForCausalLM(cfg), ids, lora_r=r, lora_alpha=alpha, lora_dropout=0.0, trainable=trainable)
    f = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
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


if __name__ == "__main__":
    run(globals())
