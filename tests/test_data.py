"""decidophobia.data + decidophobia.prompt 的测试.

跑:  PYTHONPATH=src .venv/bin/python tests/test_data.py
"""

import random

from _runner import run
from decidophobia.data import LabeledSet, MenuExample, class_split, compose_menu, menu_k_range, merge_sets
from decidophobia.prompt import render_menu, split_prompt

NAMES = {i: f"n{i}" for i in range(10)}


def _ex(query="Q", options=(0, 1), gold_idx=0, **kw):
    return MenuExample(query=query, options=list(options), gold_idx=gold_idx, label=options[gold_idx],
                       option_names=[NAMES[c] for c in options], **kw)


# --------------------------------------------------------------------------
# class_split / compose_menu
# --------------------------------------------------------------------------


def test_class_split_partitions_all_classes():
    s = class_split(n_classes=10, n_held_out=3, seed=0)
    assert len(s.train) == 7 and len(s.held_out) == 3, s
    assert sorted(s.train + s.held_out) == list(range(10)), s


def test_class_split_is_deterministic_by_seed():
    a, b, c = class_split(10, 3, seed=1), class_split(10, 3, seed=1), class_split(10, 3, seed=2)
    assert a == b and a.held_out != c.held_out


def test_compose_menu_includes_gold_and_has_k_distinct_options_from_pool():
    opts, gi = compose_menu(gold=5, pool=list(range(10)), k=4, rng=random.Random(0))
    assert len(opts) == 4 and len(set(opts)) == 4 and set(opts) <= set(range(10)) and opts[gi] == 5


def test_compose_menu_gold_position_varies_with_rng():
    rng = random.Random(0)
    positions = {compose_menu(0, list(range(6)), 3, rng)[1] for _ in range(40)}
    assert positions == {0, 1, 2}, positions


def test_compose_menu_clamps_k_to_pool_size():
    """pool 只有 2 个类 (BoolQ 的 yes/no) 时 k 夹到 2, 两个位置都要出现."""
    rng = random.Random(0)
    outs = [compose_menu(1, [0, 1], 10, rng) for _ in range(20)]
    assert all(sorted(o) == [0, 1] for o, _ in outs)
    assert {gi for _, gi in outs} == {0, 1}


def test_menu_k_range_without_k_min_means_full_menu_up_to_k_max():
    """k_min 不给 = 全量: 每个菜单都取 k_max, compose_menu 再夹到池子大小 (池子 60 就是 60 项).
    给了 k_min 才是旧的随机长度."""
    assert menu_k_range(None, 256) == (256, 256)
    assert menu_k_range(2, 10) == (2, 10)


def test_full_menu_puts_every_pool_class_in_when_pool_is_below_k_max():
    """池子 6 个类、k_max 256: 每个菜单 6 项且就是整个池子, gold 在 6 个位置都出现过."""
    s = _set(n=60, n_cls=6)
    ex = s.sample_examples(list(range(6)), menu_k_range(None, 256), 60, random.Random(0))
    assert all(sorted(e.options) == list(range(6)) for e in ex)
    assert {e.gold_idx for e in ex} == set(range(6))


# --------------------------------------------------------------------------
# LabeledSet
# --------------------------------------------------------------------------


def _set(n=30, n_cls=6, questions=None):
    return LabeledSet(queries=[f"q{i}" for i in range(n)], labels=[i % n_cls for i in range(n)],
                      names={i: f"n{i}" for i in range(n_cls)}, context_label="Customer message",
                      questions=questions)


def test_build_examples_only_uses_queries_and_menus_from_given_classes():
    s = _set(n=4, n_cls=4)
    ex = s.build_examples(classes=[1, 3], k_range=(2, 2), rng=random.Random(0))
    assert [e.query for e in ex] == ["q1", "q3"], ex
    for e in ex:
        assert set(e.options) <= {1, 3} and e.options[e.gold_idx] == e.label
        assert e.option_names == [s.names[c] for c in e.options]
        assert e.context_label == "Customer message" and e.question is None


def test_sample_examples_draws_n_with_varied_k():
    s = _set()
    ex = s.sample_examples(classes=list(range(6)), k_range=(2, 4), n=20, rng=random.Random(0))
    assert len(ex) == 20 and {len(e.options) for e in ex} <= {2, 3, 4} and len({len(e.options) for e in ex}) > 1


def test_labeled_set_qtype_flows_into_examples():
    """数据集定 qtype (choice / bool), 每条样本带着它."""
    s = _set(n=4, n_cls=2)
    assert s.qtype == "choice"
    s2 = LabeledSet(queries=s.queries, labels=s.labels, names=s.names, qtype="bool")
    ex = s2.build_examples(classes=[0, 1], k_range=(2, 2), rng=random.Random(0))
    assert all(e.qtype == "bool" for e in ex)


def test_labeled_set_carries_per_item_question():
    """BoolQ 每条各有问句; 通过 questions 列表按索引带进样本."""
    s = _set(n=3, n_cls=2, questions=["is it a?", "is it b?", "is it c?"])
    ex = s.build_examples(classes=[0, 1], k_range=(2, 2), rng=random.Random(0))
    assert [e.question for e in ex] == ["is it a?", "is it b?", "is it c?"]


def test_build_examples_draws_menu_from_pool_but_queries_from_classes():
    """留出类的题目, 菜单从留出类 + 合成意图里抽: 题只出自 classes, 干扰项可以来自 pool 的任何类."""
    s = _set(n=8, n_cls=8)
    ex = s.build_examples(classes=[1, 3], k_range=(6, 6), rng=random.Random(0), pool=[1, 3, 4, 5, 6, 7])
    assert [e.query for e in ex] == ["q1", "q3"]
    for e in ex:
        assert len(e.options) == 6 and set(e.options) <= {1, 3, 4, 5, 6, 7} and e.options[e.gold_idx] == e.label
    assert any(set(e.options) & {4, 5, 6, 7} for e in ex), "pool 里的类要真的出现在菜单里"


def test_sample_examples_k_log_favours_short_menus_but_reaches_the_top():
    """k 按对数均匀取: 2..256 之间一半的题落在 ~23 以内, 少数拉到两百多."""
    s = _set(n=300, n_cls=300)
    ex = s.sample_examples(classes=list(range(300)), k_range=(2, 256), n=250, rng=random.Random(0), k_log=True)
    ks = sorted(len(e.options) for e in ex)
    assert ks[len(ks) // 2] < 40, ks[len(ks) // 2]
    assert ks[0] == 2 and ks[-1] > 200, (ks[0], ks[-1])
    ex_u = s.sample_examples(classes=list(range(300)), k_range=(2, 256), n=250, rng=random.Random(0))
    ku = sorted(len(e.options) for e in ex_u)
    assert ku[len(ku) // 2] > 100, "默认仍是均匀"


def test_merge_sets_puts_both_in_one_id_space_and_reports_offsets():
    a = LabeledSet(queries=["a0", "a1", "a2"], labels=[0, 1, 0], names={0: "x", 1: "y"})
    b = LabeledSet(queries=["b0"], labels=[0], names={0: "z"})
    m, offsets = merge_sets(a, b)
    assert offsets == [0, 2]
    assert m.queries == ["a0", "a1", "a2", "b0"] and m.labels == [0, 1, 0, 2]
    assert m.names == {0: "x", 1: "y", 2: "z"}
    assert m.context_label == a.context_label and m.questions is None and m.qtype == "choice"


def test_merge_sets_refuses_different_context_labels_or_per_item_questions():
    a = LabeledSet(queries=["a"], labels=[0], names={0: "x"}, context_label="Customer message")
    b = LabeledSet(queries=["b"], labels=[0], names={0: "z"}, context_label="Passage")
    try:
        merge_sets(a, b)
    except ValueError:
        pass
    else:
        raise AssertionError("上下文标签不同却合并了")
    c = LabeledSet(queries=["c"], labels=[0], names={0: "w"}, questions=["q?"])
    try:
        merge_sets(a, c)
    except ValueError:
        return
    raise AssertionError("带逐条问句的集合却合并了")


# --------------------------------------------------------------------------
# prompt
# --------------------------------------------------------------------------


def test_render_menu_lists_options_with_d_tokens_in_order_and_ends_at_answer():
    s = render_menu(_ex("I lost my card", options=(7, 2), gold_idx=1))
    assert "\n<|D0|>. n7\n<|D1|>. n2\n" in s, repr(s)
    assert s.endswith("Answer:") and "<|D2|>" not in s, repr(s)


def test_context_first_puts_query_before_menu_and_splits_at_the_newline():
    ex = _ex("I lost my card", options=(7, 2), gold_idx=1)
    s = render_menu(ex, layout="context-first")
    assert s.index("I lost my card") < s.index("<|D0|>"), repr(s)
    ctx, q = split_prompt(ex, layout="context-first")
    assert ctx + q == s and ctx.endswith("\n\n") and ctx.startswith("Customer message: I lost my card")
    assert "<|D0|>" not in ctx and "I lost my card" not in q


def test_context_prefix_is_identical_across_questions():
    a = _ex("q", options=(0, 1), gold_idx=0)
    b = _ex("q", options=(3, 4, 5), gold_idx=2)
    assert split_prompt(a, layout="context-first")[0] == split_prompt(b, layout="context-first")[0]


def test_menu_first_has_no_shared_prefix():
    ctx, q = split_prompt(_ex("hello"), layout="menu-first")
    assert ctx == "" and "hello" in q and q.endswith("Answer:")


def test_question_goes_between_context_and_menu():
    """BoolQ 形状: passage 是 context, 每条自带问句, 选项 yes/no.
    context-first 下 passage 在前缀里, 问句在分支里、菜单之前."""
    ex = _ex("The sky is blue because of Rayleigh scattering.", options=(1, 0), gold_idx=0,
             context_label="Passage", question="is the sky blue because of scattering?")
    ex = MenuExample(**{**ex.__dict__, "option_names": ["yes", "no"]})
    ctx, q = split_prompt(ex, layout="context-first")
    assert ctx.startswith("Passage: The sky is blue") and "scattering?" not in ctx, repr(ctx)
    assert q.index("Question: is the sky blue") < q.index("<|D0|>. yes"), repr(q)
    assert "\n<|D0|>. yes\n<|D1|>. no\n" in q and q.endswith("Answer:"), repr(q)


def test_type_marker_sits_inside_the_question_label():
    """type_marker=True: 'Question (<|bool|>):'; False: 'Question:'. 标记跟着 Question 这个锚走, 其余一字不差."""
    ex = _ex("p", options=(1, 0), gold_idx=0, qtype="bool", question="is it?")
    on = render_menu(ex, type_marker=True)
    off = render_menu(ex, type_marker=False)
    assert "Question (<|bool|>): is it?" in on, repr(on)
    assert "Question: is it?" in off and "<|bool|>" not in off, repr(off)
    assert on.replace(" (<|bool|>)", "") == off, (on, off)


def test_type_marker_default_is_off():
    assert "<|choice|>" not in render_menu(_ex("p", qtype="choice"))


if __name__ == "__main__":
    run(globals())
