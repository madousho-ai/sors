"""各题型自己的损失: 评分题的分级距离项 (--score-emd) 与成对损失按题型过滤 (--contrast-qtypes).

评分题: 菜单行的类 id 就是级别 (0 最低). 每道评分题把菜单上的概率按级别排好, 与目标分布比累计分布,
EMD = Σ_j (CDF_p(j) - CDF_t(j))², j 取到倒数第二级. 一批的这一项 = 各评分题 EMD 之和 / 整批题数
(按题数加权: 非评分题记 0), 所以分组反传时和交叉熵一样按行数折算即可.

跑:  HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python tests/test_type_losses.py
"""

import random
from dataclasses import replace

import torch

from _runner import run
from sors.core.batch import collate, contrast_rows, group_contrasts, score_levels
from sors.core.menu import MenuExample, reorder_menu, with_partners
from sors.core.model import last_logits
from sors.training.loop import TrainConfig, backward_groups, step_loss
from sors.training.loss import score_emd_loss
from test_evaluate import D5, _hand_batch, _tiny

LEVELS = ["low", "mid", "high"]


def _score(gold, order=(0, 1, 2), target=None):
    ex = MenuExample(query="state", options=[0, 1, 2], gold_idx=gold, label=gold, option_names=LEVELS,
                     qtype="score", target=target)
    return reorder_menu(ex, list(order))


def _choice(gold=0):
    return MenuExample(query="other", options=[0, 1, 2], gold_idx=gold, label=gold, option_names=["a", "b", "c"])


def test_score_levels_give_each_menu_row_its_level_and_minus_one_elsewhere():
    exs = [_score(1, order=(2, 0, 1)), _choice()]
    got = score_levels(exs, 4)
    assert got.tolist() == [[2, 0, 1, -1], [-1, -1, -1, -1]], got


def test_emd_is_the_squared_cdf_gap_summed_over_score_rows_and_divided_by_the_batch():
    exs = [_score(2, order=(2, 0, 1)), _choice()]
    b = _hand_batch(exs)
    logits = torch.full((2, 10), -30.0)
    logits[0, D5[exs[0].slot_codes[exs[0].options.index(0)]]] = 0.0  # 全部概率压在最低级, 目标是最高级
    got = score_emd_loss(logits, b["slot_ids"], score_levels(exs, 3).to(logits.device), b["target"])
    # p = [1, 0, 0], t = [0, 0, 1]: CDF 差 [1, 1] -> 2; 一批两道题 -> 1
    assert abs(got.item() - 1.0) < 1e-5, got


def test_an_adjacent_miss_costs_less_than_a_far_one_and_a_hit_costs_nothing():
    def emd(level):
        exs = [_score(2)]
        b = _hand_batch(exs)
        logits = torch.full((1, 10), -30.0)
        logits[0, D5[exs[0].options.index(level)]] = 0.0
        return score_emd_loss(logits, b["slot_ids"], score_levels(exs, 3), b["target"]).item()
    assert emd(2) < 1e-6 < emd(1) < emd(0), (emd(2), emd(1), emd(0))


def test_soft_score_targets_are_compared_by_level():
    exs = [_score(2, order=(1, 2, 0), target=[0.0, 0.25, 0.75])]
    exs = [replace(exs[0])]
    b = _hand_batch(exs)
    b["target"] = torch.tensor([exs[0].target])
    logits = torch.full((1, 10), -30.0)
    for row, level in enumerate(exs[0].options):  # 模型给出的分布正好等于目标
        p = {0: 0.0, 1: 0.25, 2: 0.75}[level]
        logits[0, D5[exs[0].slot_codes[row]]] = torch.log(torch.tensor(max(p, 1e-12)))
    got = score_emd_loss(logits, b["slot_ids"], score_levels(exs, 3), b["target"])
    assert got.item() < 1e-6, got


def test_contrast_rows_can_be_limited_to_some_question_types():
    def pair(qtype, g):
        a = MenuExample(query="x", options=[0, 1], gold_idx=0, label=0, option_names=["no", "yes"], qtype=qtype,
                        contrast_group=g)
        return [a, replace(a, query="y", gold_idx=1, label=1)]
    exs = [*pair("bool", 1), *pair("choice", 2)]
    assert contrast_rows(exs)["group"].max().item() == 1
    only = contrast_rows(exs, {"bool"})
    assert only["row"].tolist() == [0, 1], only
    assert contrast_rows(exs, {"score"}) is None


def test_step_loss_adds_the_weighted_score_term_and_filters_pairs_by_type():
    exs = with_partners([_score(2, order=(1, 0, 2)), _choice(1)], random.Random(0))
    b = _hand_batch(exs)
    logits = torch.randn(len(exs), 10, generator=torch.Generator().manual_seed(2))
    plain = step_loss(TrainConfig(loss="menu", consistency=0.5), exs, b, logits, D5)[0]
    cfg = TrainConfig(loss="menu", consistency=0.5, score_emd=0.7)
    emd = score_emd_loss(logits, b["slot_ids"], score_levels(exs, 3), b["target"])
    assert emd.item() > 0
    total = step_loss(cfg, exs, b, logits, D5)[0]
    assert abs(total.item() - (plain.item() + 0.7 * emd.item())) < 1e-6


def test_token_budget_accumulation_matches_the_whole_batch_with_score_and_bool_only_pairs():
    tok, ids, m = _tiny()
    yes_no = ["no: it does not", "yes: it does"]
    a = MenuExample(query="the part was fitted by an approved garage", options=[0, 1], gold_idx=1, label=1,
                    option_names=yes_no, qtype="bool", contrast_group=1)
    pair = [a, replace(a, query="the part was fitted by an unknown garage", gold_idx=0, label=0)]
    exs = with_partners(group_contrasts([_score(2, order=(2, 1, 0)), pair[0], _choice(2), pair[1],
                                         _score(0, order=(0, 2, 1))]), random.Random(4))
    cfg = TrainConfig(k_max=3, loss="menu", consistency=0.5, contrast_weight=0.5, contrast_qtypes=("bool",),
                      score_emd=0.8, micro_tokens=1, accumulate_gradients=True)
    b = collate(exs, tok, ids, 3)
    m.zero_grad(set_to_none=True)
    logits = last_logits(m, b["input_ids"], b["attention_mask"])
    loss, ce, js = step_loss(cfg, exs, b, logits, ids)
    loss.backward()
    expected = {k: p.grad.clone() for k, p in m.named_parameters() if p.grad is not None}
    m.zero_grad(set_to_none=True)
    backward_groups(m, cfg, exs, b, ids, b["attention_mask"].sum(1).tolist())
    for k, p in m.named_parameters():
        if k in expected:
            torch.testing.assert_close(p.grad, expected[k], rtol=2e-4, atol=2e-5)


if __name__ == "__main__":
    run(globals())
