"""scripts/baseline-banking77.py 里纯函数的测试.

跑:  .venv/bin/pytest tests/test_banking77_metrics.py
每个期望值都是手算的, 写在各自的 docstring 里.
"""

import importlib.util
import math
import pathlib

_SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "baseline-banking77.py"
_spec = importlib.util.spec_from_file_location("baseline_banking77", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


class _FakeTok:
    """只把给定集合里的字符串编成单 token, 其余一律两个 token. id 值本身无意义."""

    def __init__(self, singles):
        self.singles = set(singles)

    def encode(self, s, add_special_tokens=False):
        return [abs(hash(s)) % 100000] if s in self.singles else [1, 2]


# --------------------------------------------------------------------------
# 码 token 选取
# --------------------------------------------------------------------------


def test_pick_code_tokens_requires_both_variants_single():
    """码在答案位置是 ' XY' (带空格), 在菜单行里是 'XY' (不带), 两个 id 都要是单 token.

    候选 'QX' 两种写法都单 -> 入选; 'ZQ' 只有带空格的单 -> 剔除.
    """
    tok = _FakeTok({" QX", "QX", " ZQ"})
    got = _mod.pick_code_tokens(tok, n=1, seed=0)
    assert got == ["QX"], got


def test_pick_code_tokens_excludes_word_like_codes():
    """小写与首字母大写形式也是单 token 的码 (AI/AS/AT...) 是英文词, 带内容无关的先验.

    'AI' 四种写法全单 -> 剔除; 'QX' 只有大写两种单 -> 入选.
    """
    tok = _FakeTok({" AI", "AI", " ai", " Ai", " QX", "QX"})
    got = _mod.pick_code_tokens(tok, n=1, seed=0)
    assert got == ["QX"], got


def test_pick_code_tokens_is_deterministic_by_seed():
    """同一 seed 两次调用给同一列表; 池子大于 n 时不同 seed 应给出不同子集."""
    pool = {p + c for c in ["QX", "ZQ", "XQ", "QZ", "ZX", "XZ"] for p in ["", " "]}
    tok = _FakeTok(pool)
    a = _mod.pick_code_tokens(tok, n=3, seed=7)
    b = _mod.pick_code_tokens(tok, n=3, seed=7)
    c = _mod.pick_code_tokens(tok, n=3, seed=8)
    assert a == b, (a, b)
    assert a != c, (a, c)


def test_pick_code_tokens_raises_when_pool_too_small():
    """池子只有 1 个合格码却要 2 个, 必须报错而非静默少给."""
    tok = _FakeTok({" QX", "QX"})
    try:
        _mod.pick_code_tokens(tok, n=2, seed=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


# --------------------------------------------------------------------------
# 码 <-> 意图 的分配
# --------------------------------------------------------------------------


def test_assign_codes_is_a_bijection_deterministic_by_seed():
    """每个 label 恰得一个码, 每个码恰用一次; 同 seed 同结果; 不同 seed 应给不同置换."""
    labels = ["a", "b", "c", "d"]
    codes = ["QX", "ZQ", "XQ", "QZ"]
    m1 = _mod.assign_codes(labels, codes, seed=1)
    m2 = _mod.assign_codes(labels, codes, seed=1)
    m3 = _mod.assign_codes(labels, codes, seed=2)
    assert m1 == m2, (m1, m2)
    assert sorted(m1) == labels and sorted(m1.values()) == sorted(codes), m1
    assert m1 != m3, (m1, m3)


# --------------------------------------------------------------------------
# 模板
# --------------------------------------------------------------------------


def test_humanize_label_strips_underscores_case_and_punctuation():
    """原始 label 里有两处异形: 'Refund_not_showing_up' 首字母大写, 'reverted_card_payment?' 带问号."""
    assert _mod.humanize_label("Refund_not_showing_up") == "refund not showing up"
    assert _mod.humanize_label("reverted_card_payment?") == "reverted card payment"
    assert _mod.humanize_label("card_arrival") == "card arrival"


def test_menu_prompt_puts_codes_at_line_start_and_ends_with_colon():
    """菜单行里码不带前导空格 ('\\nQX. '), 答案位置的延续带前导空格 (' QX').

    两处 token id 不同, 模型没法直接从上下文拷贝那个 id. 提示以冒号收尾,
    所以读出用带空格的变体 —— 与 BoolQ 补全式模板同一条约定.
    """
    s = _mod.render_menu_prompt("I lost my card", {"card_arrival": "QX", "card_linking": "ZQ"})
    assert "\nQX. card arrival\n" in s, repr(s)
    assert "\nZQ. card linking\n" in s, repr(s)
    assert "I lost my card" in s, repr(s)
    assert s.endswith(":"), repr(s[-30:])


def test_menu_prompt_lists_labels_in_canonical_order():
    """菜单顺序固定为 label 字母序, 与置换无关 —— 置换只换码, 不换行序."""
    s = _mod.render_menu_prompt("q", {"card_linking": "QX", "card_arrival": "ZQ"})
    assert s.index("ZQ. card arrival") < s.index("QX. card linking"), repr(s)


def test_menu_prompt_with_empty_query_keeps_message_slot():
    """空 query 用于量先验 (contextual calibration): 菜单照给, 消息行留空, 其余一字不差."""
    full = _mod.render_menu_prompt("hello", {"a": "QX"})
    empty = _mod.render_menu_prompt("", {"a": "QX"})
    assert full.replace("hello", "") == empty, (full, empty)


def test_bare_prompt_has_no_menu():
    """裸模板给 h_bare 用: 只有用户句和 'Intent:' 结尾, 不含任何码或意图名."""
    s = _mod.render_bare_prompt("I lost my card")
    assert "I lost my card" in s and s.endswith(":"), repr(s)
    assert "QX" not in s and "card arrival" not in s, repr(s)


# --------------------------------------------------------------------------
# 多类指标
# --------------------------------------------------------------------------


def test_renormalize_k_scales_each_row_to_one():
    """[[0.2, 0.2], [0.1, 0.3]] -> [[0.5, 0.5], [0.25, 0.75]]."""
    got = _mod.renormalize_k([[0.2, 0.2], [0.1, 0.3]])
    want = [[0.5, 0.5], [0.25, 0.75]]
    for g, w in zip(got, want):
        for a, b in zip(g, w):
            assert abs(a - b) < 1e-12, (got, want)


def test_topk_accuracy_counts_label_within_top_k():
    """行 0 正确类 2 排第 1; 行 1 正确类 0 排第 2; 行 2 正确类 1 排第 3.

    top-1 = 1/3, top-2 = 2/3, top-3 = 3/3.
    """
    q = [[0.1, 0.2, 0.7], [0.3, 0.6, 0.1], [0.5, 0.1, 0.4]]
    y = [2, 0, 1]
    assert abs(_mod.topk_accuracy(q, y, k=1) - 1 / 3) < 1e-12
    assert abs(_mod.topk_accuracy(q, y, k=2) - 2 / 3) < 1e-12
    assert abs(_mod.topk_accuracy(q, y, k=3) - 1.0) < 1e-12


def test_brier_multiclass_sums_squared_error_over_classes():
    """约定 mean_i sum_k (q_ik - y_ik)^2 (无 1/K, 二元情形下是 brier_binary 的 2 倍).

    行 0: q=[0.9, 0.1], y=0 -> 0.01 + 0.01 = 0.02
    行 1: q=[0.2, 0.8], y=0 -> 0.64 + 0.64 = 1.28
    均值 0.65
    """
    got = _mod.brier_multiclass([[0.9, 0.1], [0.2, 0.8]], [0, 0])
    assert abs(got - 0.65) < 1e-12, got


def test_calibrate_by_prior_divides_then_renormalizes():
    """contextual calibration: q_k / prior_k 再归一.

    q=[0.5, 0.5], prior=[0.8, 0.2] -> [0.625, 2.5] -> [0.2, 0.8].
    均匀先验不改变 q.
    """
    got = _mod.calibrate_by_prior([[0.5, 0.5]], [0.8, 0.2])
    assert abs(got[0][0] - 0.2) < 1e-12 and abs(got[0][1] - 0.8) < 1e-12, got
    same = _mod.calibrate_by_prior([[0.3, 0.7]], [0.5, 0.5])
    assert abs(same[0][0] - 0.3) < 1e-12 and abs(same[0][1] - 0.7) < 1e-12, same


def test_nll_multiclass_reads_correct_class_column():
    """q=[[0.9, 0.1], [0.2, 0.8]], y=[0, 1] -> -(ln0.9 + ln0.8)/2 = 0.1642520330."""
    got = _mod.nll_multiclass([[0.9, 0.1], [0.2, 0.8]], [0, 1])
    assert abs(got - 0.1642520330) < 1e-9, got


def test_nll_multiclass_clamps_saturated_zero():
    """正确类概率精确为 0 时返回有限值: [1.0, 0.0] -> -(ln1 + ln1e-12)/2 = 13.815510558."""
    got = _mod.nll_multiclass([[1.0, 0.0], [0.0, 1.0]], [0, 0])
    assert abs(got - 13.815510558) < 1e-9, got


def test_ece_multiclass_uses_max_prob_and_argmax():
    """conf 取每行 max, 对错取 argmax==y, 然后走等宽分箱.

    行 0: max 0.6 (类 0), y=0 对; 行 1: max 0.7 (类 1), y=0 错 -> 落 [0.50,0.75)
    行 2: max 0.8 (类 0), y=0 对; 行 3: max 0.9 (类 1), y=1 对 -> 落 [0.75,1.00]
    箱 1: 均值 0.65, 准确率 0.5, 差 0.15, 权 2/4; 箱 2: 均值 0.85, 准确率 1.0, 差 0.15, 权 2/4
    ECE = 0.15
    """
    q = [[0.6, 0.4], [0.3, 0.7], [0.8, 0.2], [0.1, 0.9]]
    got = _mod.ece_multiclass(q, [0, 0, 0, 1], n_bins=4)
    assert abs(got - 0.15) < 1e-12, got


# --------------------------------------------------------------------------
# 产物路径
# --------------------------------------------------------------------------


def test_result_paths_keep_dots_in_model_id_and_encode_run():
    """文件名带模式与置换 seed, 模型名里的小数点不得被截断."""
    j, n = _mod.result_paths("results", "Qwen/Qwen3-1.7B-Base", "menu", 3, 3080, "20260921-200000")
    want = "results/20260921-200000-banking77-Qwen_Qwen3-1.7B-Base-menu-perm3-n3080"
    assert str(j) == want + ".json", str(j)
    assert str(n) == want + ".npz", str(n)
