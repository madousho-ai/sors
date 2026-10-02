"""sors.evaluation.metrics 里二元指标的测试 + BoolQ adapter.

跑:  PYTHONPATH=src .venv/bin/python tests/test_binary.py
"""

from _runner import run
from sors.core.menu import MenuExample
from sors.evaluation.metrics import auroc, binary_summary


def test_auroc_counts_correctly_ordered_pairs():
    """scores [0.1, 0.4, 0.35, 0.8], labels [0, 0, 1, 1].
    正例 0.35 只赢负例 0.1 (1/2), 正例 0.8 赢两个 (2/2) -> 3/4 = 0.75."""
    assert abs(auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) - 0.75) < 1e-12


def test_auroc_perfect_and_constant():
    assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]) == 0.5  # 平局记半分


def test_auroc_is_none_when_one_class_absent():
    assert auroc([0.1, 0.9], [1, 1]) is None


def _ex(options, gold_idx):
    return MenuExample(query="q", options=list(options), gold_idx=gold_idx, label=options[gold_idx],
                       option_names=["yes" if c == 1 else "no" for c in options])


def test_binary_summary_maps_position_probs_back_to_class_space():
    """三条: yes 分别在位置 0 / 1 / 0, gold 分别是 yes / yes / no.
    q = [[0.7,0.3],[0.2,0.8],[0.6,0.4]] -> P(yes) = [0.7, 0.8, 0.6], y = [1, 1, 0]
    yes_rate (P(yes) >= 0.5 的比例) = 3/3; auroc: 负例 0.6 vs 正例 0.7, 0.8 -> 1.0
    brier_binary = mean((p - y)^2) = (0.09 + 0.04 + 0.36) / 3 = 0.163333
    """
    exs = [_ex((1, 0), 0), _ex((0, 1), 1), _ex((1, 0), 1)]
    q = [[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]]
    m = binary_summary(q, exs, pos_class=1)
    assert abs(m["p_pos_mean"] - 0.7) < 1e-12, m
    assert m["pos_rate"] == 1.0 and m["label_pos_rate"] == 2 / 3, m
    assert m["auroc"] == 1.0, m
    assert abs(m["brier_binary"] - 0.163333333) < 1e-9, m


def test_boolq_adapter_shapes():
    """google/boolq validation 3270 条, 类 0=no 1=yes, context_label Passage, 每条有问句."""
    from sors.data.boolq import load_boolq

    tr, va = load_boolq()
    assert len(va.queries) == 3270 and len(tr.queries) == 9427, (len(tr.queries), len(va.queries))
    assert va.names == {0: "no", 1: "yes"} and va.context_label == "Passage"
    assert len(va.questions) == 3270 and va.questions[0].endswith("?")
    assert abs(sum(va.labels) / len(va.labels) - 0.6217) < 0.001


if __name__ == "__main__":
    run(globals())
