"""scoring.evaluate 与 training.loop 的评估点的测试. 一层、hidden 16 的随机 Qwen3 配真的 tokenizer, 走真的 prepare_model / collate, CPU 上跑.

跑:  HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python tests/test_evaluate.py
"""

import random

import torch

from _runner import run
from sors.core.batch import collate, pair_alignment
from sors.core.menu import MenuExample, arrangements, with_partners
from sors.core.model import grouped_last_logits, last_logits, prepare_model
from sors.core.prompt import DEFAULT_LAYOUT
from sors.core.tokens import install_d_tokens, install_type_tokens
from sors.evaluation.metrics import first_two_slots, pass_consistency
from sors.evaluation.scoring import EvalSet, consistency_eval, evaluate, score_examples
from sors.training.loop import TrainConfig, eval_record, probe_passes, step_loss, step_target, train
from sors.training.loss import consistency_js, menu_hits, training_loss

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


class _Writer:
    """SummaryWriter 的替身, 只记 add_scalar."""

    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def flush(self):
        pass


def test_train_reports_accuracy_on_the_training_batch_and_leaves_out_questions_without_an_answer():
    """一步训练: 这一步的 logits 就是训练前的模型算的, 期望值先用 menu_hits 算好.
    批里第三道是均匀标签 (文本没提), 不进分母. TensorBoard 记 train/accuracy, log.jsonl 那条记 train_accuracy
    (上一个评估点以来的全部训练题) 与它的题数 train_accuracy_n; step 0 还没训练, 两样是 None / 0."""
    tok, d_ids, m = _tiny()
    names = ["change pin", "top up", "card lost"]
    exs = [MenuExample(query="I lost my card", options=[0, 1, 2], gold_idx=2, label=2, option_names=names),
           MenuExample(query="my top up failed", options=[0, 1, 2], gold_idx=1, label=1, option_names=names),
           MenuExample(query="hello", options=[0, 1, 2], gold_idx=0, label=0, option_names=names, target=[1 / 3] * 3)]
    b = collate(exs, tok, d_ids, k_max=3)
    with torch.no_grad():
        hits, n = menu_hits(last_logits(m, b["input_ids"], b["attention_mask"]), b["slot_ids"], b["target"])
    assert n == 2

    w = _Writer()
    hist = train(m, tok, d_ids, lambda n_, rng: list(exs), {},
                 TrainConfig(steps=1, batch_size=3, k_max=3, loss="vocab", eval_every=1, log_every=1), writer=w)
    assert [(t, s) for t, _, s in w.scalars if t == "train/accuracy"] == [("train/accuracy", 1)], w.scalars
    assert [v for t, v, _ in w.scalars if t == "train/accuracy"] == [hits / n]
    assert [(r["step"], r["train_accuracy"], r["train_accuracy_n"]) for r in hist] == [(0, None, 0), (1, hits / n, 2)]


def test_train_accuracy_in_the_log_covers_every_step_since_the_previous_eval_point():
    """eval_every 3 / log_every 1: step 3 那条的 train_accuracy_n 是三步的题数合计, 不是最后一步的."""
    tok, d_ids, m = _tiny()
    ex = MenuExample(query="I lost my card", options=[0, 1, 2], gold_idx=2, label=2,
                     option_names=["change pin", "top up", "card lost"])
    hist = train(m, tok, d_ids, lambda n_, rng: [ex] * n_, {},
                 TrainConfig(steps=3, batch_size=2, k_max=3, loss="vocab", eval_every=3, log_every=1))
    assert [(r["step"], r["train_accuracy_n"]) for r in hist] == [(0, 0), (3, 6)]


def _uneven_questions():
    """四道长短差得很远的题, 批里长短交错."""
    names = ["change pin", "top up", "card lost"]
    queries = ["hi", "my card was lost on the train this morning and I need a new one sent to my home address "
               "as soon as possible please", "top up", "the top up I made yesterday evening never arrived"]
    return [MenuExample(query=q, options=[0, 1, 2], gold_idx=i % 3, label=i % 3, option_names=names)
            for i, q in enumerate(queries)]


def test_grouped_last_logits_give_each_row_what_one_forward_over_the_whole_batch_gives():
    """分两组各自前向, 每组只补齐到组里最长; 放回原来的行序后与整批一次前向逐行相同 (浮点误差内),
    对 logits 求的梯度也相同. 一组就是整批原样一次前向, 逐位相同.
    梯度的量级在几十, 批形状不同时求和顺序不同, 所以按相对误差比."""
    tok, d_ids, m = _tiny()
    b = collate(_uneven_questions(), tok, d_ids, k_max=3)
    whole = last_logits(m, b["input_ids"], b["attention_mask"])
    split = grouped_last_logits(m, b["input_ids"], b["attention_mask"], 2)
    assert torch.allclose(split, whole, atol=1e-6), (split - whole).abs().max()
    assert torch.equal(grouped_last_logits(m, b["input_ids"], b["attention_mask"], 1), whole)

    weights = torch.randn(whole.shape, generator=torch.Generator().manual_seed(0))
    grads = []
    for logits_fn in (lambda: last_logits(m, b["input_ids"], b["attention_mask"]),
                      lambda: grouped_last_logits(m, b["input_ids"], b["attention_mask"], 2)):
        m.zero_grad(set_to_none=True)
        (logits_fn() * weights).sum().backward()
        grads.append({n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None})
    assert grads[0].keys() == grads[1].keys() and len(grads[0]) == 9, grads[0].keys()
    for n in grads[0]:
        assert torch.allclose(grads[0][n], grads[1][n], rtol=1e-4, atol=1e-4 * grads[0][n].abs().max()), n


def test_train_runs_each_length_group_through_the_model_on_its_own_width():
    """micro_batches 2: 一步四道题分两次前向, 每次两道; 短的那组只有它自己最长那条的宽度.
    这一步的训练 loss 与整批一次前向的相同."""
    exs = _uneven_questions()
    lengths = collate(exs, *_tiny()[:2], k_max=3)["attention_mask"].sum(1).tolist()
    losses = {}
    for mb in (1, 2):
        tok, d_ids, m = _tiny()
        shapes = []
        m.get_input_embeddings().register_forward_pre_hook(lambda mod, args: shapes.append(tuple(args[0].shape)))
        hist = train(m, tok, d_ids, lambda n_, rng: list(exs), {},
                     TrainConfig(steps=1, batch_size=4, k_max=3, loss="vocab", eval_every=1, log_every=1,
                                 micro_batches=mb))
        losses[mb] = hist[-1]["train_loss"]
        if mb == 1:
            assert shapes == [(4, max(lengths))], shapes
        else:
            ranked = sorted(lengths, reverse=True)
            assert shapes == [(2, ranked[0]), (2, ranked[2])], (shapes, lengths)
    assert abs(losses[1] - losses[2]) < 1e-5, losses


def test_train_groups_rows_by_padded_token_budget():
    """micro_tokens > 0 时按 batch.token_groups 分组, 不看 micro_batches: 每组的形状是 (行数, 组里最长),
    行数 × 最长不超过预算. 整步一次反传与逐组反传两条路的 loss 都与整批一次前向相同."""
    from sors.core.batch import token_groups

    exs = _uneven_questions()
    lengths = collate(exs, *_tiny()[:2], k_max=3)["attention_mask"].sum(1).tolist()
    budget = 2 * sorted(lengths, reverse=True)[1]
    expected = [(len(g), max(lengths[r] for r in g)) for g in token_groups(lengths, budget)]
    assert 1 < len(expected) < len(exs), expected
    losses = {}
    for name, extra in (("whole", {"micro_batches": 1}),
                        ("tokens", {"micro_batches": 1, "micro_tokens": budget}),
                        ("tokens-accumulate", {"micro_batches": 1, "micro_tokens": budget,
                                               "accumulate_gradients": True})):
        tok, d_ids, m = _tiny()
        shapes = []
        m.get_input_embeddings().register_forward_pre_hook(lambda mod, args: shapes.append(tuple(args[0].shape)))
        hist = train(m, tok, d_ids, lambda n_, rng: list(exs), {},
                     TrainConfig(steps=1, batch_size=4, k_max=3, loss="vocab", eval_every=1, log_every=1, **extra))
        losses[name] = hist[-1]["train_loss"]
        if name != "whole":
            assert shapes == expected, (name, shapes, expected)
    assert abs(losses["whole"] - losses["tokens"]) < 1e-5, losses
    assert abs(losses["whole"] - losses["tokens-accumulate"]) < 1e-5, losses


if __name__ == "__main__":
    run(globals())
