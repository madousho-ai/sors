"""train.evaluate 的测试. 一层、hidden 16 的随机 Qwen3 配真的 tokenizer, 走真的 prepare_model / collate, CPU 上跑.

跑:  HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python tests/test_evaluate.py
"""

import torch

from _runner import run
from decidophobia.batch import collate
from decidophobia.data import MenuExample
from decidophobia.metrics import first_two_slots
from decidophobia.model import last_logits, prepare_model
from decidophobia.prompt import DEFAULT_LAYOUT
from decidophobia.tokens import install_d_tokens, install_type_tokens
from decidophobia.train import EvalSet, evaluate, score_examples, step_target

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


if __name__ == "__main__":
    run(globals())
