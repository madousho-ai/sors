"""train.evaluate 的测试. 一层、hidden 16 的随机 Qwen3 配真的 tokenizer, 走真的 prepare_model / collate, CPU 上跑.

跑:  HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python tests/test_evaluate.py
"""

import random

import torch

from _runner import run
from decidophobia.batch import collate, pair_alignment
from decidophobia.data import MenuExample, arrangements, with_partners
from decidophobia.loss import consistency_js, training_loss
from decidophobia.metrics import first_two_slots, pass_consistency
from decidophobia.model import last_logits, prepare_model
from decidophobia.prompt import DEFAULT_LAYOUT
from decidophobia.tokens import install_d_tokens, install_type_tokens
from decidophobia.train import (EvalSet, TrainConfig, consistency_eval, eval_record, evaluate, probe_passes,
                                score_examples, step_loss, step_target)

MODEL = "Qwen/Qwen3-0.6B-Base"


def _tiny():
    """词表与 Qwen3 同大 (151936, D 码落在空行里), 其余缩到一层 hidden 16."""
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    tok = AutoTokenizer.from_pretrained(MODEL)
    d_ids = install_d_tokens(tok)
    ids = d_ids + install_type_tokens(tok)
    cfg = Qwen3Config(vocab_size=151936, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=8)
    torch.manual_seed(0)
    m = prepare_model(Qwen3ForCausalLM(cfg), ids, lora_r=4, lora_alpha=8, lora_dropout=0.0)
    return tok, d_ids, m


def test_evaluate_reports_the_cross_entropy_over_the_whole_vocabulary():
    """vocab_ce 与 --loss vocab 的训练 loss 同一个分母: 整个词表. 每题取正确选项绑的那个 D 码
    (连续编号下菜单第 j 项是 <|Dj|>) 在全词表 log_softmax 里的值, 取负再平均.
    未训练的模型 D 码只分到很小一点概率, 只在菜单上归一的 nll 与它差得很远."""
    tok, d_ids, m = _tiny()
    exs = [
        MenuExample(query="I lost my card", options=[0, 1, 2], gold_idx=2, label=2,
                    option_names=["change pin", "top up", "card lost"]),
        MenuExample(query="my top up failed", options=[0, 1], gold_idx=0, label=0,
                    option_names=["top up failed", "card lost"]),
    ]
    r = evaluate(m, tok, d_ids, EvalSet(exs, batch_size=2), k_max=3, max_length=512, layout=DEFAULT_LAYOUT)

    b = collate(exs, tok, d_ids, k_max=3)  # 与 evaluate 同一批, 左填充一样
    m.eval()
    with torch.no_grad():
        logp = torch.log_softmax(last_logits(m, b["input_ids"], b["attention_mask"]), -1)
    want = -(logp[0, d_ids[2]] + logp[1, d_ids[0]]).item() / 2
    assert abs(r["vocab_ce"] - want) < 1e-5, (r["vocab_ce"], want)
    assert r["vocab_ce"] > r["nll"] + 1, (r["vocab_ce"], r["nll"])


def test_evaluate_reports_how_much_lands_on_the_first_two_slots():
    """evaluate 报 first_two_slots 的三个量, 与拿同一批逐题概率直接算的相同. 两题正确答案在 D2 / D0: gold 率 1/2."""
    tok, d_ids, m = _tiny()
    exs = [
        MenuExample(query="I lost my card", options=[0, 1, 2], gold_idx=2, label=2,
                    option_names=["change pin", "top up", "card lost"]),
        MenuExample(query="my top up failed", options=[0, 1], gold_idx=0, label=0,
                    option_names=["top up failed", "card lost"]),
    ]
    es = EvalSet(exs, batch_size=2)
    r = evaluate(m, tok, d_ids, es, k_max=3, max_length=512, layout=DEFAULT_LAYOUT)
    s = score_examples(m, tok, d_ids, exs, 2, 3, 512, DEFAULT_LAYOUT)
    want = first_two_slots(s["q"], s["gold"])
    assert r["gold_d01_rate"] == 0.5, r["gold_d01_rate"]
    assert all(abs(r[k] - v) < 1e-9 for k, v in want.items()), ({k: r.get(k) for k in want}, want)


def test_step_target_is_none_for_hard_labels_without_smoothing_so_the_old_loss_runs():
    """一批全是硬标签、平滑 0: 不给 target, 训练走按 gold 下标的老损失, 旧 run 逐位复现.
    有一条带软标签、或平滑大于 0: 给整批的目标分布 (硬标签那几行是 one-hot), 平滑摊在各自菜单上."""
    hard = MenuExample(query="a", options=[0, 1, 2], gold_idx=2, label=2, option_names=["x", "y", "z"])
    soft = MenuExample(query="b", options=[0, 1], gold_idx=0, label=0, option_names=["x", "y"], target=[0.75, 0.25])
    b = {"target": torch.tensor([[0.0, 0.0, 1.0], [0.75, 0.25, 0.0]]), "slot_ids": torch.tensor([[5, 6, 7], [5, 6, -1]])}
    assert step_target([hard, hard], b, 0.0) is None
    assert step_target([hard, soft], b, 0.0) is b["target"]
    got = step_target([hard, hard], b, 0.3)
    assert torch.allclose(got, torch.tensor([[0.1, 0.1, 0.8], [0.675, 0.325, 0.0]])), got


D5 = [5, 6, 7, 8, 9]  # 手搭的 batch: 词表 10, D 码在 id 5..9


def _hand_batch(exs):
    """collate 会给的 slot_ids / gold / target, 不经 tokenizer."""
    k = max(len(e.options) for e in exs)
    slot_ids = torch.full((len(exs), k), -1)
    target = torch.zeros(len(exs), k)
    for i, e in enumerate(exs):
        slot_ids[i, : len(e.options)] = torch.tensor([D5[c] for c in e.slot_codes])
        target[i, e.gold_idx] = 1.0
    return {"slot_ids": slot_ids, "gold": torch.tensor([e.gold_idx for e in exs]), "target": target}


def _two_questions():
    return [MenuExample(query="a", options=[0, 1, 2], gold_idx=2, label=2, option_names=["x", "y", "z"]),
            MenuExample(query="b", options=[3, 4], gold_idx=0, label=3, option_names=["no", "yes"], qtype="bool")]


def test_step_loss_without_consistency_is_the_training_loss_alone():
    """consistency 0: 一致性项不算, 优化的就是原来的 training_loss, 旧 run 逐位复现."""
    exs = _two_questions()
    b, logits = _hand_batch(exs), torch.randn(2, 10, generator=torch.Generator().manual_seed(0))
    total, ce, js = step_loss(TrainConfig(loss="vocab"), exs, b, logits, D5)
    assert js is None and total is ce
    assert ce.item() == training_loss("vocab", logits, b["slot_ids"], b["gold"], D5).item()


def test_step_loss_with_consistency_adds_lambda_times_the_js_of_each_adjacent_pair():
    """batch 是 with_partners 排好的 [a, a', b, b']: 交叉熵照旧对四条取平均, 再加 λ · 两对 JS 的平均."""
    exs = with_partners(_two_questions(), random.Random(0))
    b, logits = _hand_batch(exs), torch.randn(4, 10, generator=torch.Generator().manual_seed(1))
    total, ce, js = step_loss(TrainConfig(loss="vocab", consistency=0.5), exs, b, logits, D5)
    want_js = consistency_js(logits, b["slot_ids"], pair_alignment(exs, b["slot_ids"].shape[1]))
    assert abs(js.item() - want_js.item()) < 1e-7 and js.item() > 0, (js, want_js)
    assert ce.item() == training_loss("vocab", logits, b["slot_ids"], b["gold"], D5).item()
    assert abs(total.item() - (ce.item() + 0.5 * js.item())) < 1e-6, (total, ce, js)


def test_step_loss_with_consistency_refuses_a_batch_that_is_not_in_pairs():
    exs = _two_questions()  # 两道不同的题挨着, 不是同一道题的两种排法
    try:
        step_loss(TrainConfig(loss="vocab", consistency=0.5), exs, _hand_batch(exs), torch.zeros(2, 10), D5)
    except ValueError:
        return
    raise AssertionError("two different questions were paired")


def _questions(n):
    names = ["change pin", "top up", "card lost", "refund"]
    return [MenuExample(query=f"message {i}", options=[0, 1, 2], gold_idx=i % 3, label=i % 3, option_names=names[:3])
            for i in range(n)]


def test_probe_passes_are_a_fixed_subset_in_fixed_random_arrangements():
    """从评估集里抽 size 道 (够不上就全部, 顺序照旧), 排成 passes 种随机的样子; 同一个 key 永远给同一套,
    于是整场训练每个评估点比的都是同一批题、同样的排法."""
    exs = _questions(10)
    got = probe_passes(exs, 4, 5, "probe-0-banking77")
    assert len(got) == 5 and all(len(p) == 4 for p in got)
    assert all(p[i].query == got[0][i].query for p in got for i in range(4))
    assert got == probe_passes(exs, 4, 5, "probe-0-banking77")
    assert got != probe_passes(exs, 4, 5, "probe-0-massive")
    everything = probe_passes(exs[:3], 4, 2, "k")
    assert [e.query for e in everything[0]] == [e.query for e in exs[:3]]


def test_consistency_eval_scores_each_pass_and_compares_them_by_description():
    tok, d_ids, m = _tiny()
    passes = arrangements(_questions(3), 3, random.Random(0))
    got = consistency_eval(m, tok, d_ids, passes, batch_size=2, k_max=3, max_length=512, layout=DEFAULT_LAYOUT)
    qs = [score_examples(m, tok, d_ids, p, 2, 3, 512, DEFAULT_LAYOUT)["q"] for p in passes]
    want = pass_consistency(qs, passes)
    assert got.keys() == want.keys() and got["n"] == 3 and got["passes"] == 3, got
    assert all(abs(got[k] - want[k]) < 1e-6 for k in want), (got, want)


def test_eval_record_reports_only_probe_consistency_until_the_final_step():
    """训练中的评估点只报探针子集上的一致性 (accuracy / agree / js). 最后一步再加全量评估集的两样:
    eval 是部署形态 (连续编号) 下原来那套正确率指标, consistency_full 是全量题的同一种一致性."""
    tok, d_ids, m = _tiny()
    binary = [MenuExample(query=f"passage {i}", options=[0, 1], gold_idx=i % 2, label=i % 2, option_names=["no", "yes"],
                          qtype="bool") for i in range(4)]
    eval_sets = {"intents": EvalSet(_questions(5), batch_size=4), "boolq": EvalSet(binary, batch_size=4, pos_class=1)}
    cfg = TrainConfig(k_max=3, max_length=512, probe_size=2, probe_passes=3)
    probes = {name: probe_passes(es.examples, cfg.probe_size, cfg.probe_passes, name) for name, es in eval_sets.items()}
    mid = eval_record(m, tok, d_ids, eval_sets, probes, cfg, final=False)
    assert mid.keys() == {"consistency"}, mid.keys()
    assert {name: (r["n"], r["passes"]) for name, r in mid["consistency"].items()} == {"intents": (2, 3), "boolq": (2, 3)}
    end = eval_record(m, tok, d_ids, eval_sets, probes, cfg, final=True)
    assert end.keys() == {"consistency", "eval", "consistency_full"}, end.keys()
    assert end["eval"]["intents"]["n"] == 5 and "auroc" in end["eval"]["boolq"]
    assert {name: r["n"] for name, r in end["consistency_full"].items()} == {"intents": 5, "boolq": 4}


if __name__ == "__main__":
    run(globals())
