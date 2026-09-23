"""scripts/choice-loglik-banking77.py 里纯函数的测试.

零依赖 (numpy 除外), 直接跑:  .venv/bin/python tests/test_choice_loglik.py
每个期望值都是手算的, 写在各自的 docstring 里.
"""

import importlib.util
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _runner import run  # noqa: E402

_SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "choice-loglik-banking77.py"
_spec = importlib.util.spec_from_file_location("choice_loglik_banking77", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

NAN = float("nan")


# --------------------------------------------------------------------------
# 提示
# --------------------------------------------------------------------------


def test_render_prefix_lists_every_choice_and_ends_with_answer_colon():
    """用户句在前, 问句, 'Choices:' 下每行 '- 名字', 空行, 'Answer:' 收尾 (冒号后无空格,
    选项续写自带前导空格)."""
    got = _mod.render_prefix("my card hasn't arrived", ["card arrival", "lost or stolen card"])
    want = (
        "Customer message: my card hasn't arrived\n\n"
        "Question: Which option best describes the message?\n"
        "Choices:\n- card arrival\n- lost or stolen card\n\n"
        "Answer:"
    )
    assert got == want, repr(got)


def test_render_prefix_without_menu_drops_only_the_choices_block():
    """bare 对照: 用户句与问句原样, 只去掉 Choices 块."""
    got = _mod.render_prefix("my card hasn't arrived", ["card arrival"], with_menu=False)
    want = "Customer message: my card hasn't arrived\n\nQuestion: Which option best describes the message?\nAnswer:"
    assert got == want, repr(got)


# --------------------------------------------------------------------------
# 逐 token logprob -> 每个选项一个分数
# --------------------------------------------------------------------------

# 1 条样本, 2 个选项. 选项 0 = 两个名字 token + 终止符; 选项 1 = 一个名字 token + 终止符, 余位 NaN
_LP = np.array([[[-1.0, -2.0, -0.5], [-3.0, -0.25, NAN]]])


def test_aggregate_sum_with_terminator_adds_every_valid_token():
    """选项 0: -1 -2 -0.5 = -3.5; 选项 1: -3 -0.25 = -3.25."""
    got = _mod.aggregate(_LP, how="sum", terminator=True)
    assert np.allclose(got, [[-3.5, -3.25]]), got


def test_aggregate_sum_without_terminator_drops_last_valid_token():
    """去掉每行最后一个有效 token (终止符): 选项 0: -1 -2 = -3; 选项 1: -3."""
    got = _mod.aggregate(_LP, how="sum", terminator=False)
    assert np.allclose(got, [[-3.0, -3.0]]), got


def test_aggregate_avg_divides_by_counted_tokens():
    """带终止符: -3.5/3, -3.25/2; 不带: -3/2 = -1.5, -3/1 = -3."""
    with_t = _mod.aggregate(_LP, how="avg", terminator=True)
    without = _mod.aggregate(_LP, how="avg", terminator=False)
    assert np.allclose(with_t, [[-3.5 / 3, -1.625]]), with_t
    assert np.allclose(without, [[-1.5, -3.0]]), without


def test_aggregate_rejects_unknown_mode():
    try:
        _mod.aggregate(_LP, how="max", terminator=True)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


# --------------------------------------------------------------------------
# restricted softmax
# --------------------------------------------------------------------------


def test_restricted_softmax_normalizes_over_the_choices():
    """[0, ln3] -> [1/4, 3/4]."""
    got = _mod.restricted_softmax(np.array([[0.0, math.log(3)]]))
    assert np.allclose(got, [[0.25, 0.75]]), got


def test_restricted_softmax_divides_by_temperature():
    """[0, 2ln3] 在 T=2 下等于 [0, ln3] -> [1/4, 3/4]."""
    got = _mod.restricted_softmax(np.array([[0.0, 2 * math.log(3)]]), T=2.0)
    assert np.allclose(got, [[0.25, 0.75]]), got


def test_restricted_softmax_is_stable_for_very_negative_scores():
    """整体平移不改结果: [-1000, -1000+ln3] 仍是 [1/4, 3/4], 不下溢成 NaN."""
    got = _mod.restricted_softmax(np.array([[-1000.0, -1000.0 + math.log(3)]]))
    assert np.allclose(got, [[0.25, 0.75]]), got


# --------------------------------------------------------------------------
# 菜单位置
# --------------------------------------------------------------------------


def test_position_mass_averages_probability_at_each_menu_line():
    """两条, K=2. 条 0 的顺序 [0,1]: 位置 0 是类 0 (0.9); 条 1 的顺序 [1,0]: 位置 0 是类 1 (0.8).
    位置 0 的平均 = (0.9+0.8)/2 = 0.85, 位置 1 = (0.1+0.2)/2 = 0.15."""
    q = np.array([[0.9, 0.1], [0.2, 0.8]])
    order = np.array([[0, 1], [1, 0]])
    got = _mod.position_mass(q, order)
    assert np.allclose(got, [0.85, 0.15]), got


# --------------------------------------------------------------------------
# 跨菜单顺序的稳定性
# --------------------------------------------------------------------------


def test_pairwise_agreement_reports_argmax_agreement_and_total_variation():
    """三套顺序给出的分布 (已映回类空间), 两条样本:
    a = [[.6,.4],[.3,.7]], b = [[.7,.3],[.6,.4]], c = [[.6,.4],[.3,.7]]
    argmax: a=[0,1] b=[0,0] c=[0,1]. 两两一致: ab 1/2, ac 2/2, bc 1/2 -> 平均 2/3.
    TV = ½Σ|差|: ab 条0 0.1 条1 0.3 -> 0.2; ac 0; bc 0.2 -> 平均 0.4/3."""
    a = np.array([[0.6, 0.4], [0.3, 0.7]])
    b = np.array([[0.7, 0.3], [0.6, 0.4]])
    got = _mod.pairwise_agreement([a, b, a.copy()])
    assert math.isclose(got["argmax_agree"], 2 / 3), got
    assert math.isclose(got["tv_mean"], 0.4 / 3), got


def test_entropy_bits_of_uniform_and_point_mass():
    """均匀 4 路 = 2 bit; 点质量 = 0 (0·log0 记 0)."""
    got = _mod.entropy_bits(np.array([[0.25] * 4, [1.0, 0.0, 0.0, 0.0]]))
    assert np.allclose(got, [2.0, 0.0]), got


# --------------------------------------------------------------------------
# 温度
# --------------------------------------------------------------------------


def test_fit_temperature_recovers_analytic_optimum():
    """四条同分 [0, 1], 三条标 1 一条标 0. NLL 在 σ(1/T) = 3/4 处最小, 即 T = 1/ln3 ≈ 0.9102."""
    s = np.array([[0.0, 1.0]] * 4)
    y = np.array([1, 1, 1, 0])
    got = _mod.fit_temperature(s, y)
    assert abs(got - 1 / math.log(3)) < 1e-3, got


def test_fit_temperature_goes_high_when_scores_carry_no_signal():
    """分数说 1 而标签一半 0 一半 1: 最好的做法是把分布摊平, T 走到搜索上界附近 (>= 50)."""
    s = np.array([[0.0, 1.0]] * 4)
    y = np.array([1, 0, 1, 0])
    got = _mod.fit_temperature(s, y)
    assert got >= 50, got


def test_heldout_temperature_fits_on_other_fold_and_applies_to_this_one():
    """按下标奇偶分两折. 标签 [1,1,1,1,1,1,0,0]: 偶数折 (0,2,4,6) 与奇数折 (1,3,5,7) 都是三个 1 一个 0,
    两折各自的最优 T 都是 1/ln3, 应用到对方后每行都是 [1/4, 3/4]."""
    s = np.array([[0.0, 1.0]] * 8)
    y = np.array([1, 1, 1, 1, 1, 1, 0, 0])
    q, temps = _mod.heldout_temperature(s, y, folds=2)
    assert len(temps) == 2 and all(abs(t - 1 / math.log(3)) < 1e-3 for t in temps), temps
    assert np.allclose(q, [[0.25, 0.75]] * 8, atol=1e-4), q


# --------------------------------------------------------------------------
# 长度偏置
# --------------------------------------------------------------------------


def test_length_profile_groups_classes_by_name_token_count():
    """三个类, 名字长度 [1, 2, 2]. q = [[.5,.3,.2],[.1,.6,.3]], y = [1, 2].
    长度 1: 1 个类, 标签占比 0, 质量占比 (.5+.1)/2 = .3, argmax [0,1] 里有 1 个落在长度 1 -> .5
    长度 2: 2 个类, 标签占比 1, 质量占比 (.5+.9)/2 = .7, argmax 占比 .5"""
    q = np.array([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]])
    got = _mod.length_profile(q, np.array([1, 2]), np.array([1, 2, 2]))
    assert [r["len"] for r in got] == [1, 2], got
    r1, r2 = got
    assert r1["n_classes"] == 1 and r2["n_classes"] == 2, got
    assert math.isclose(r1["label_share"], 0.0) and math.isclose(r2["label_share"], 1.0), got
    assert math.isclose(r1["mass_share"], 0.3) and math.isclose(r2["mass_share"], 0.7), got
    assert math.isclose(r1["pred_share"], 0.5) and math.isclose(r2["pred_share"], 0.5), got


if __name__ == "__main__":
    run(globals())
