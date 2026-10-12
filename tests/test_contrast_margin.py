"""成对对数几率差损失 (--contrast-weight / --contrast-margin) 的测试.

原题与改动题 (contrast_group 相同的两道题) 只差一处事实、答案相反. 设两道题的答案是 A、B, 每一行算
d = log p(B) - log p(A); 损失 softplus(m - (d 在答 B 的行上的平均 - d 在答 A 的行上的平均)), 按组平均.
两份材料共有的套路特征在差里抵消, 梯度只能来自那处改动.

跑:  HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python tests/test_contrast_margin.py
"""

import math
import random
from dataclasses import replace

import torch
import torch.nn.functional as F

from _runner import run
from sors.core.batch import collate, contrast_rows, group_contrasts, token_groups
from sors.core.menu import MenuExample, with_partners
from sors.core.model import last_logits
from sors.training.loop import TrainConfig, backward_groups, contrast_term, step_loss
from sors.training.loss import contrast_margin_loss
from test_evaluate import D5, _hand_batch, _tiny

OPTS = ["approve", "decline", "ask"]


def _q(query, label, group, target=None):
    return MenuExample(query=query, options=[0, 1, 2], gold_idx=label, label=label, option_names=OPTS,
                       target=target, contrast_group=group)


def _pair(group=7):
    return [_q("fitted by an approved garage", 0, group), _q("fitted by an unapproved garage", 1, group)]


def test_contrast_rows_find_each_group_and_the_menu_rows_of_both_answers():
    exs = [_q("other", 2, None), *_pair(7)]
    exs = [exs[0], replace(exs[2], options=[2, 1, 0], option_names=OPTS[::-1], gold_idx=1), exs[1]]
    rows = contrast_rows(exs)
    # 组 7 的答案按第一次出现排: A = 1 (改动题先出现), B = 0
    assert rows["row"].tolist() == [1, 2] and rows["group"].tolist() == [0, 0], rows
    assert rows["is_b"].tolist() == [False, True], rows
    assert rows["pos_a"].tolist() == [1, 1] and rows["pos_b"].tolist() == [2, 0], rows


def test_groups_with_the_same_answer_soft_labels_or_a_missing_row_are_skipped():
    same = [_q("x", 0, 1), _q("y", 0, 1)]
    soft = [_q("x", 0, 2, target=[0.7, 0.3, 0.0]), _q("y", 1, 2)]
    lone = [_q("x", 0, 3)]
    missing = [_q("x", 0, 4), replace(_q("y", 1, 4), options=[1, 2], option_names=OPTS[1:], gold_idx=0)]
    assert contrast_rows([*same, *soft, *lone, *missing]) is None


def test_the_loss_is_softplus_of_the_margin_minus_the_log_odds_gap():
    exs = _pair()
    b = _hand_batch(exs)
    logits = torch.zeros(2, 10)
    logits[0, D5[0]], logits[0, D5[1]] = 2.0, 0.5   # 原题 (答 A=0): d = 0.5 - 2.0 = -1.5
    logits[1, D5[0]], logits[1, D5[1]] = 1.0, 1.5   # 改动题 (答 B=1): d = 0.5
    got = contrast_margin_loss(logits, b["slot_ids"], contrast_rows(exs), margin=2.0)
    assert abs(got.item() - F.softplus(torch.tensor(2.0 - (0.5 - -1.5))).item()) < 1e-6, got


def test_answering_both_items_alike_is_penalised_and_its_gradient_separates_them():
    """模型忽略改动、两道题给同样的 logits 时差为 0, 损失 softplus(m); 梯度把改动题推向 B, 原题推向 A."""
    exs = _pair()
    b = _hand_batch(exs)
    logits = torch.zeros(2, 10, requires_grad=True)
    loss = contrast_margin_loss(logits, b["slot_ids"], contrast_rows(exs), margin=2.0)
    assert abs(loss.item() - math.log1p(math.exp(2.0))) < 1e-6
    loss.backward()
    g = logits.grad
    assert g[1, D5[1]] < 0 < g[1, D5[0]] and g[0, D5[0]] < 0 < g[0, D5[1]], g


def test_consistency_copies_count_on_their_own_side_of_the_pair():
    """with_partners 之后每道题两份; 四行按各自的答案分到两边取平均, 与只算原题两份时一样."""
    exs = with_partners(_pair(), random.Random(0))
    b = _hand_batch(exs)
    logits = torch.randn(4, 10, generator=torch.Generator().manual_seed(3))
    rows = contrast_rows(exs)
    assert rows["row"].tolist() == [0, 1, 2, 3] and rows["is_b"].tolist() == [False, False, True, True]
    got = contrast_margin_loss(logits, b["slot_ids"], rows, margin=1.0)
    lp = torch.log_softmax(torch.stack([logits[i, b["slot_ids"][i]] for i in range(4)]), 1)
    d = [lp[i, exs[i].options.index(1)] - lp[i, exs[i].options.index(0)] for i in range(4)]
    want = F.softplus(1.0 - ((d[2] + d[3]) / 2 - (d[0] + d[1]) / 2))
    assert abs(got.item() - want.item()) < 1e-6, (got, want)


def test_grouping_puts_the_members_of_a_contrast_group_next_to_each_other_without_reordering_the_rest():
    a, b, c, d = (_q(t, 0, None) for t in "abcd")
    x1, x2 = _pair(5)
    y1, y2 = _pair(9)
    got = group_contrasts([a, x1, b, y2, c, x2, d, y1])
    assert got == [a, x1, x2, b, y2, y1, c, d], [e.query for e in got]


def test_token_groups_never_split_a_span():
    lengths = [9, 1, 1, 8, 2, 2, 7, 3]
    spans = [1, 2, 1, 2, 2]  # 行 [0] [1,2] [3] [4,5] [6,7]
    groups = token_groups(lengths, 18, spans)
    whole = [[0], [1, 2], [3], [4, 5], [6, 7]]
    for g in groups:
        for s in whole:
            assert set(s) <= set(g) or not set(s) & set(g), (g, s)
    assert sorted(r for g in groups for r in g) == list(range(8))
    for g in groups:
        assert len(g) * max(lengths[r] for r in g) <= 18 or len(g) <= 2, groups


def test_step_loss_adds_the_weighted_contrast_term():
    exs = with_partners([_q("other", 2, None), *_pair()], random.Random(1))
    b = _hand_batch(exs)
    logits = torch.randn(len(exs), 10, generator=torch.Generator().manual_seed(4))
    plain = step_loss(TrainConfig(loss="vocab", consistency=0.5), exs, b, logits, D5)[0]
    cfg = TrainConfig(loss="vocab", consistency=0.5, contrast_weight=0.3, contrast_margin=2.0)
    total = step_loss(cfg, exs, b, logits, D5)[0]
    term = contrast_term(cfg, exs, b, logits)
    assert term is not None and term.item() > 0
    assert abs(total.item() - (plain.item() + 0.3 * term.item())) < 1e-6
    assert contrast_term(TrainConfig(loss="vocab"), exs, b, logits) is None


def test_token_budget_accumulation_keeps_contrast_pairs_together_and_matches_whole_batch_gradients():
    tok, ids, m = _tiny()
    exs = with_partners(group_contrasts([_q("an unrelated short one", 2, None), _pair(1)[0],
                                         _q("another unrelated question that is a little longer", 1, None),
                                         _pair(1)[1], *_pair(2)]), random.Random(5))
    cfg = TrainConfig(k_max=3, loss="vocab", consistency=0.5, contrast_weight=0.7, contrast_margin=2.0,
                      micro_tokens=1, accumulate_gradients=True)
    b = collate(exs, tok, ids, 3)
    m.zero_grad(set_to_none=True)
    logits = last_logits(m, b["input_ids"], b["attention_mask"])
    loss, ce, js = step_loss(cfg, exs, b, logits, ids)
    loss.backward()
    expected = {k: p.grad.clone() for k, p in m.named_parameters() if p.grad is not None}
    want_margin = contrast_term(cfg, exs, b, logits).item()
    m.zero_grad(set_to_none=True)
    lengths = b["attention_mask"].sum(1).tolist()
    got_ce, got_js, _, _, got_margin, _ = backward_groups(m, cfg, exs, b, ids, lengths)
    assert abs(got_ce - ce.item()) < 2e-5 and abs(got_js - js.item()) < 2e-6
    assert abs(got_margin - want_margin) < 2e-5, (got_margin, want_margin)
    for k, p in m.named_parameters():
        if k in expected:
            torch.testing.assert_close(p.grad, expected[k], rtol=2e-4, atol=2e-5)


if __name__ == "__main__":
    run(globals())
