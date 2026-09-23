"""decidophobia.tokens / batch / loss / metrics 的测试.

跑:  PYTHONPATH=src .venv/bin/python tests/test_batch_loss.py
"""

import math
import sys

import torch

from decidophobia.batch import collate
from decidophobia.data import MenuExample
from decidophobia.loss import LOSSES, all_slot_cross_entropy, answer_mass, slot_cross_entropy, training_loss
from decidophobia.metrics import (answer_mass_summary, brier_multiclass, by_gold_slot, ece_multiclass, menu_size_summary,
                                  nll_multiclass, topk_accuracy)
from decidophobia.tokens import D_TOKENS, TYPE_TOKENS, install_d_tokens, install_type_tokens
from decidophobia.train import scalar_items

MODEL = "Qwen/Qwen3-0.6B-Base"


def _tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL)


# --------------------------------------------------------------------------
# tokens
# --------------------------------------------------------------------------


def test_d_tokens_are_256_named_slots():
    assert len(D_TOKENS) == 256 and D_TOKENS[0] == "<|D0|>" and D_TOKENS[255] == "<|D255|>", D_TOKENS[:2]


def test_install_d_tokens_fills_spare_rows_without_resize():
    """Qwen3 tokenizer 151669 个, vocab_size 151936, 空 267 行. 256 个 D-token 应落在 151669..151924."""
    tok = _tok()
    ids = install_d_tokens(tok)
    assert ids == list(range(151669, 151669 + 256)), (ids[0], ids[-1])
    assert ids[-1] < 151936


def test_install_d_tokens_is_idempotent():
    """同一 tokenizer 装两次给同一批 id (add_tokens 第二次返回 0, 不能因此返回空表)."""
    tok = _tok()
    a = install_d_tokens(tok)
    b = install_d_tokens(tok)
    assert a == b and len(b) == 256


def test_d_token_is_single_token_right_after_colon():
    """'Answer:<|D3|>' 必须切成 [..., ':', <|D3|>], 中间没有空格 token —— 读出位置是冒号之后紧邻的那一位."""
    tok = _tok()
    ids = install_d_tokens(tok)
    enc = tok.encode("Answer:<|D3|>", add_special_tokens=False)
    assert enc[-1] == ids[3], tok.convert_ids_to_tokens(enc)
    assert tok.convert_ids_to_tokens(enc[-2]) == ":", tok.convert_ids_to_tokens(enc)


def test_type_tokens_take_the_rows_after_d_tokens():
    """<|choice|> <|bool|> <|score|> 紧跟 D255 之后 (151925..151927), 仍在 vocab_size 151936 内."""
    tok = _tok()
    d = install_d_tokens(tok)
    t = install_type_tokens(tok)
    assert TYPE_TOKENS == ["<|choice|>", "<|bool|>", "<|score|>"]
    assert t == [d[-1] + 1, d[-1] + 2, d[-1] + 3] and t[-1] < 151936, t


def test_type_marker_tokenizes_cleanly_inside_parentheses():
    """'Question (<|bool|>): is it?' 必须切成 [..., ' (', <|bool|>, '):', ...]:
    特殊 token 两侧的括号各自独立, 不与 Question 或问句合并."""
    tok = _tok()
    install_d_tokens(tok)
    t = install_type_tokens(tok)
    ids = tok.encode("Question (<|bool|>): is it?", add_special_tokens=False)
    toks = tok.convert_ids_to_tokens(ids)
    i = ids.index(t[1])
    assert toks[i - 1] == "Ġ(" and toks[i + 1] == "):", toks
    assert toks[0] == "Question", toks


# --------------------------------------------------------------------------
# collate
# --------------------------------------------------------------------------


def test_collate_left_pads_so_answer_position_is_last():
    """两条长短不一的提示, 左填充后每行最后一个 token 都是 ':'; 短的那条前面是 pad."""
    tok = _tok()
    d_ids = install_d_tokens(tok)
    exs = [
        MenuExample(query="short", options=[0, 1], gold_idx=0, label=0, option_names=["a", "b"]),
        MenuExample(query="a much longer customer message here", options=[2, 0, 1], gold_idx=2, label=1,
                    option_names=["c", "a", "b"]),
    ]
    b = collate(exs, tok, d_ids, k_max=4)
    colon = tok.encode(":", add_special_tokens=False)[0]
    assert b["input_ids"].shape[0] == 2
    assert (b["input_ids"][:, -1] == colon).all(), b["input_ids"][:, -1]
    assert b["attention_mask"][0, 0] == 0 and b["attention_mask"][1, 0] == 1, b["attention_mask"][:, 0]


def test_collate_builds_slot_ids_by_position_and_pads_with_minus_one():
    """slot_ids[i, j] = D_j 的 id (j < k_i), 其余 -1; gold 是位置索引."""
    tok = _tok()
    d_ids = install_d_tokens(tok)
    exs = [
        MenuExample(query="x", options=[5, 9], gold_idx=1, label=9, option_names=["n5", "n9"]),
        MenuExample(query="y", options=[1, 2, 3], gold_idx=0, label=1, option_names=["n1", "n2", "n3"]),
    ]
    b = collate(exs, tok, d_ids, k_max=4)
    assert b["slot_ids"].tolist() == [
        [d_ids[0], d_ids[1], -1, -1],
        [d_ids[0], d_ids[1], d_ids[2], -1],
    ], b["slot_ids"]
    assert b["gold"].tolist() == [1, 0]


def test_collate_slot_ids_follow_each_examples_codes():
    """菜单绑的是 D10 / D233 时, slot_ids 那一行就是 [D10, D233], gold 仍是位置 ->
    loss 的目标是 D233 这个 token, 与 all-slots 的分母对得上."""
    tok = _tok()
    d_ids = install_d_tokens(tok)
    ex = MenuExample(query="x", options=[5, 9], gold_idx=1, label=9, option_names=["n5", "n9"], codes=[10, 233])
    b = collate([ex], tok, d_ids, k_max=3)
    assert b["slot_ids"].tolist() == [[d_ids[10], d_ids[233], -1]], b["slot_ids"]
    assert b["gold"].tolist() == [1]
    assert all_slot_cross_entropy(torch.zeros(1, 151936), b["slot_ids"], b["gold"], d_ids).item() > 0


# --------------------------------------------------------------------------
# loss
# --------------------------------------------------------------------------


def test_slot_cross_entropy_masks_padding_and_averages():
    """V=10.
    行 0: 槽 [5,6,7] 的 logit [1,2,3], gold=2 -> -ln(e^3/(e+e^2+e^3)) = 0.40760596
    行 1: 槽 [8,9,-1], logit [0,0,(第 3 位是 pad, 其 id 位置 logit 设 99 也不得参与)], gold=0 -> ln 2 = 0.69314718
    mean = 0.55037657
    """
    logits = torch.zeros(2, 10)
    logits[0, 5], logits[0, 6], logits[0, 7] = 1.0, 2.0, 3.0
    logits[1, 8], logits[1, 9] = 0.0, 0.0
    logits[1, 0] = 99.0  # pad 槽 clamp 到 id 0 之后若没 mask 会读到它
    slot_ids = torch.tensor([[5, 6, 7], [8, 9, -1]])
    gold = torch.tensor([2, 0])
    got = slot_cross_entropy(logits, slot_ids, gold).item()
    assert abs(got - 0.55037657) < 1e-6, got


def test_all_slot_cross_entropy_puts_every_d_slot_in_the_denominator():
    """V=10, 4 个 D-token 在 id 5..8. 分母是这 4 个, 不论菜单几项; 非 D 的 token 不进分母.
    行 0: 菜单 [5,6,pad] (k=2), gold=1 -> 目标 id 6. logit id5..8 = [1,2,3,0], 菜单外的 id 7 也进分母;
          id 0 = 99 不是 D, 不得参与. -ln(e²/(e+e²+e³+1)) = 1.44018970
    行 1: 菜单 [5,6,7], gold=2 -> 目标 id 7, D 上 logit 全 0 -> ln 4 = 1.38629436
    mean = 1.41324203
    """
    logits = torch.zeros(2, 10)
    logits[0, 5], logits[0, 6], logits[0, 7] = 1.0, 2.0, 3.0
    logits[:, 0] = 99.0
    slot_ids = torch.tensor([[5, 6, -1], [5, 6, 7]])
    gold = torch.tensor([1, 2])
    got = all_slot_cross_entropy(logits, slot_ids, gold, d_ids=[5, 6, 7, 8]).item()
    assert abs(got - 1.41324203) < 1e-6, got


def test_all_slot_cross_entropy_rejects_a_gold_slot_outside_d_ids():
    """菜单第 gold 位的 token 不在 d_ids 里时报错, 不能静默把目标当成第 0 个 D."""
    logits = torch.zeros(1, 10)
    slot_ids = torch.tensor([[5, 9]])
    try:
        all_slot_cross_entropy(logits, slot_ids, torch.tensor([1]), d_ids=[5, 6, 7, 8])
    except ValueError:
        return
    raise AssertionError("gold slot 9 is not a D id, expected ValueError")


def test_training_loss_dispatches_on_kind():
    """'menu' 是只在菜单 k 个槽上归一的老损失, 'all-slots' 是全部 D 槽; 其他名字报错."""
    logits = torch.zeros(2, 10)
    logits[0, 5], logits[0, 6], logits[0, 7] = 1.0, 2.0, 3.0
    slot_ids = torch.tensor([[5, 6, -1], [5, 6, 7]])
    gold = torch.tensor([1, 2])
    d_ids = [5, 6, 7, 8]
    menu = training_loss("menu", logits, slot_ids, gold, d_ids).item()
    full = training_loss("all-slots", logits, slot_ids, gold, d_ids).item()
    assert menu == slot_cross_entropy(logits, slot_ids, gold).item(), menu
    assert full == all_slot_cross_entropy(logits, slot_ids, gold, d_ids).item(), full
    assert menu != full
    assert LOSSES == ("menu", "all-slots")
    try:
        training_loss("vocab", logits, slot_ids, gold, d_ids)
    except ValueError:
        return
    raise AssertionError("unknown loss kind, expected ValueError")


def test_answer_mass_splits_full_vocab_into_menu_offmenu_and_rest():
    """V=10, 4 个 D-token 在 id 5..8. 全词表 softmax 分三块: 菜单里的 k 个槽 / 菜单外的 D 槽 / 其余.
    行 0: 菜单 [5,6,pad], 其余 logit 0, id 3 = 2 -> Z = 9+e², 菜单 2/Z, 菜单外 (7,8) 2/Z, top1 = 3 不在菜单.
          pad 若没 mask 会 clamp 到 id 0 多算 1/Z.
    行 1: 菜单 [5,6,7], id 6 = 2 -> 菜单 (2+e²)/Z, 菜单外只剩 8: 1/Z, top1 = 6 在菜单里.
    """
    e2 = math.e**2
    z = 9 + e2
    logits = torch.zeros(2, 10)
    logits[0, 3] = 2.0
    logits[1, 6] = 2.0
    slot_ids = torch.tensor([[5, 6, -1], [5, 6, 7]])
    m, off, top1 = answer_mass(logits, slot_ids, d_ids=[5, 6, 7, 8])
    assert torch.allclose(m, torch.tensor([2 / z, (2 + e2) / z])), m
    assert torch.allclose(off, torch.tensor([2 / z, 1 / z])), off
    assert top1.tolist() == [False, True], top1


def test_answer_mass_summary_uses_baseline_names_and_numpy_percentiles():
    """键名与 baseline 脚本的 m_answer_* 一致, 基模与训练后可以并排比. 分位数用 numpy 默认的线性插值:
    [.1 .2 .3 .4 .5] 的 p05 在下标 0.2 -> 0.12, p95 在下标 3.8 -> 0.48."""
    got = answer_mass_summary([0.3, 0.1, 0.5, 0.2, 0.4], [0.0, 0.1, 0.0, 0.0, 0.4], [True, False, True, True, False])
    want = {"m_answer_mean": 0.3, "m_answer_p05": 0.12, "m_answer_p95": 0.48,
            "m_offmenu_mean": 0.1, "top1_in_menu_rate": 0.6}
    assert got.keys() == want.keys(), got
    assert all(abs(got[k] - v) < 1e-12 for k, v in want.items()), got


# --------------------------------------------------------------------------
# metrics (与 baseline-banking77.py 同一套手算期望)
# --------------------------------------------------------------------------


def test_topk_accuracy():
    q = [[0.1, 0.2, 0.7], [0.3, 0.6, 0.1], [0.5, 0.1, 0.4]]
    y = [2, 0, 1]
    assert abs(topk_accuracy(q, y, k=1) - 1 / 3) < 1e-12
    assert abs(topk_accuracy(q, y, k=2) - 2 / 3) < 1e-12


def test_nll_multiclass_clamps():
    assert abs(nll_multiclass([[0.9, 0.1], [0.2, 0.8]], [0, 1]) - 0.1642520330) < 1e-9
    assert abs(nll_multiclass([[1.0, 0.0]], [1]) - 27.631021115) < 1e-6


def test_brier_multiclass():
    assert abs(brier_multiclass([[0.9, 0.1], [0.2, 0.8]], [0, 0]) - 0.65) < 1e-12


def test_ece_multiclass():
    q = [[0.6, 0.4], [0.3, 0.7], [0.8, 0.2], [0.1, 0.9]]
    assert abs(ece_multiclass(q, [0, 0, 0, 1], n_bins=4) - 0.15) < 1e-12


def _onehot_row(n, hot, p=0.9):
    """长度 n, 第 hot 位 p, 其余均分剩下的."""
    rest = (1 - p) / (n - 1)
    return [p if i == hot else rest for i in range(n)]


def test_by_gold_slot_bins_every_ten_slots_and_skips_empty_bins():
    """正确答案在 D0 / D3 / D12 / D25 / D27. 模型答: D0 对, D3 错 (答 D1), D12 对, D25 错 (答 D0), D27 对.
    0-9: n 2 acc 0.5; 10-19: n 1 acc 1.0; 20-29: n 2 acc 0.5. 30 以后没有题, 不出现."""
    n = 30
    q = [_onehot_row(n, 0), _onehot_row(n, 1), _onehot_row(n, 12), _onehot_row(n, 0), _onehot_row(n, 27)]
    y = [0, 3, 12, 25, 27]
    got = by_gold_slot(q, y, width=10)
    assert list(got) == ["0-9", "10-19", "20-29"], list(got)
    assert [got[b]["n"] for b in got] == [2, 1, 2], got
    assert [got[b]["accuracy"] for b in got] == [0.5, 1.0, 0.5], got
    assert abs(got["10-19"]["nll"] - (-math.log(0.9))) < 1e-12, got["10-19"]
    assert set(got["0-9"]) == {"n", "accuracy", "top5_accuracy", "nll", "conf_mean"}, got["0-9"]


def test_menu_size_summary_reports_min_max_mean():
    assert menu_size_summary([10, 60, 60, 2]) == {"k_min": 2, "k_max": 60, "k_mean": 33.0}


def test_scalar_items_flattens_nested_dicts_and_skips_none():
    """TensorBoard 只收标量: 嵌套的 by_gold_slot 拍平成 a/b/c, None 丢掉."""
    got = scalar_items("eval/seen", {"accuracy": 0.5, "auroc": None,
                                     "by_gold_slot": {"0-9": {"n": 2, "accuracy": 1.0}}})
    assert got == [("eval/seen/accuracy", 0.5), ("eval/seen/by_gold_slot/0-9/n", 2),
                   ("eval/seen/by_gold_slot/0-9/accuracy", 1.0)], got


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{failed} failed")
    sys.exit(1 if failed else 0)
