"""scripts/baseline-boolq.py 里纯函数的测试.

跑:  .venv/bin/pytest tests/test_baseline_metrics.py
每个期望值都是手算的, 写在各自的 docstring 里.
"""

import importlib.util
import math
import pathlib

_SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "baseline-boolq.py"
_spec = importlib.util.spec_from_file_location("baseline_boolq", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_renormalize_binary_splits_mass_proportionally():
    """P(yes)=0.30, P(no)=0.10 -> 分母 0.40 -> 0.75 / 0.25."""
    q_yes, q_no = _mod.renormalize_binary(0.30, 0.10)
    assert abs(q_yes - 0.75) < 1e-12, q_yes
    assert abs(q_no - 0.25) < 1e-12, q_no


def test_prob_from_logsumexp_reconstructs_softmax_exactly():
    """只存 logit 和全向量的 logsumexp, 应当能还原出与直接 softmax 相同的概率."""
    logits = [2.0, -1.0, 0.5, 3.25, -4.0]
    lse = math.log(sum(math.exp(z) for z in logits))
    direct = [math.exp(z) / sum(math.exp(w) for w in logits) for z in logits]
    recon = [_mod.prob_from_logsumexp(z, lse) for z in logits]
    for d, r in zip(direct, recon):
        assert abs(d - r) < 1e-12, (d, r)


def test_nll_matches_hand_computation():
    """正确类概率 [0.9, 0.8] -> -(ln0.9 + ln0.8)/2 = 0.1642520330."""
    got = _mod.nll([0.9, 0.8])
    assert abs(got - 0.1642520330) < 1e-9, got


def test_nll_clamps_probability_saturated_to_zero():
    """概率在浮点里饱和到精确的 0 时应返回有限值.

    Qwen3-1.7B/chat 在 BoolQ 上实际触发过: 答错的那条 q_correct == 0.0,
    math.log(0) 抛 ValueError: math domain error, 整次跑作废.
    夹到 eps=1e-12 后 [1.0, 0.0] -> -(ln1 + ln1e-12)/2 = 13.815510558.
    """
    got = _mod.nll([1.0, 0.0])
    assert abs(got - 13.815510558) < 1e-9, got


def test_count_saturated_counts_entries_below_eps():
    """[1.0, 0.0, 1e-13, 1e-11] 里低于 eps=1e-12 的是 0.0 和 1e-13, 共 2 条."""
    got = _mod.count_saturated([1.0, 0.0, 1e-13, 1e-11])
    assert got == 2, got


def test_brier_binary_matches_hand_computation():
    """q_yes=[0.9, 0.2], y=[1, 0] -> (0.1^2 + 0.2^2)/2 = 0.025."""
    got = _mod.brier_binary([0.9, 0.2], [1, 0])
    assert abs(got - 0.025) < 1e-12, got


def test_ece_weights_bins_by_count():
    """4 个样本, 4 等宽分箱.

    conf 0.6/0.7 落 [0.50,0.75): 均值 0.65, 准确率 0.5, 差 0.15, 权重 2/4
    conf 0.8/0.9 落 [0.75,1.00]: 均值 0.85, 准确率 1.0, 差 0.15, 权重 2/4
    ECE = 0.5*0.15 + 0.5*0.15 = 0.15
    """
    got = _mod.ece([0.6, 0.7, 0.8, 0.9], [False, True, True, True], n_bins=4)
    assert abs(got - 0.15) < 1e-12, got


def test_ece_ignores_empty_bins():
    """全部样本落进同一个箱时, 空箱不得贡献数值, 结果应为 |均值 conf - 准确率|.

    conf 0.8/0.9 均落 [0.75,1.00], 均值 0.85, 准确率 0.5 -> 0.35
    """
    got = _mod.ece([0.8, 0.9], [True, False], n_bins=4)
    assert abs(got - 0.35) < 1e-12, got


def test_slot_table_keeps_only_single_token_variants():
    """多 token 的写法必须被剔除, 否则读出会悄悄取到首个子词的 logit.

    'uncertain' 是 [1347, 7615] 两个 token, 不得入表;
    ' uncertain' 是 [35118] 单 token, 应当入表.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    table = _mod.build_slot_table(tok, {"uncertain": ["uncertain", " uncertain"]})
    assert "uncertain" not in table["uncertain"], table["uncertain"]
    assert table["uncertain"][" uncertain"] == 35118, table["uncertain"]


def test_completion_template_ends_with_colon():
    """补全式模板以冒号结尾, 延续因此带前导空格 —— 读出必须用空格变体.

    这个断言把「模板末尾字符」和「该读哪个 token id」这层耦合钉住.
    """
    s = _mod.render_prompt("passage-question-oneword-v1", None, "PASSAGE", "QUESTION")
    assert s.endswith(":"), repr(s[-40:])


def test_chat_template_closes_thinking_block():
    """Qwen3 instruct 默认开思考模式, 提示停在 assistant 头, 下一个 token 是 <think>.

    读出必须发生在答案位置, 所以渲染结果必须已经把 think 块闭合掉.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    s = _mod.render_prompt("chat-oneword-v1", tok, "PASSAGE", "QUESTION")
    assert s.endswith("</think>\n\n"), repr(s[-60:])


def test_result_paths_keep_dots_in_model_id():
    """模型名含小数点时文件名不得被截断, 且必须带上 template id.

    Path("...Qwen3-0.6B-Base-n200").with_suffix(".json") 会把最后一个点之后的
    ".6B-Base-n200" 当成扩展名替换掉, 得到 "...Qwen3-0.json".
    """
    j, n = _mod.result_paths("results", "Qwen/Qwen3-0.6B-Base", "chat-oneword-v1", 200, "20260921-164010")
    want = "results/20260921-164010-boolq-Qwen_Qwen3-0.6B-Base-chat-oneword-v1-n200"
    assert str(j) == want + ".json", str(j)
    assert str(n) == want + ".npz", str(n)
