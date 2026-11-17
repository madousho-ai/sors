"""ladder_split / ladder_pool: 按数据量阶梯找每类的饱和点.

每类先留 eval_n 道题 (按材料组) 作为这一类自己的留出题, 永不训练; 其余材料组排好固定顺序.
一档给每类一个 n: 取这一类顺序里前面的组, 直到绑定数 >= n. 小 n 的选择是大 n 的前缀.

跑: uv run pytest -q tests/test_synth_ladder.py
"""
import collections

from sors.data.synth_v5 import V5Item, family_of, ladder_pool, ladder_split

CLASSES = {"base": ["base"], "compute": ["xrq"], "tg": ["xtg"]}


def _it(domain, ctx, q="q1", goal="breadth", contrast=None):
    return V5Item(domain=domain, context_id=ctx, question_id=q, kind="k", goal=goal, label="", texts=["t"], asks=["a"],
                  rows=["x", "y"], bare=None, qtype="choice", target=None, gold=0, contrast=contrast)


def _items():
    out = []
    for m in range(200):
        out.append(_it("banking", f"cancel#{m}/b-m{m}", "q1"))
        out.append(_it("banking", f"cancel#{m}/b-m{m}", "q2"))
    for m in range(100):
        out.append(_it("banking", f"banking_xrq/rq_a/m{m}", goal="complex"))
    for m in range(60):
        out.append(_it("tg_spam", f"tg_spam_xtg/tg_spam_a/m{m}"))
    src = len(out) - 1
    out.append(_it("tg_spam", "tg_spam_xtg/tg_spam_a/m59e", goal="contrast", contrast=src))
    return out


def test_every_class_holds_out_its_own_eval_bindings_by_material_group():
    sp = ladder_split(_items(), CLASSES, eval_n=20)
    items = sp["items"]
    for c in CLASSES:
        ev = sp["eval"][c]
        assert 20 <= len(ev) <= 21 and {family_of(items[i]) for i in ev} == set(CLASSES[c])
        held = {items[i].context_id for i in ev}
        train = {items[i].context_id for g in sp["order"][c] for i in g}
        assert not held & train


def test_held_out_materials_are_dropped_with_their_edits():
    sp = ladder_split(_items(), CLASSES, eval_n=10, drop={"tg_spam_xtg/tg_spam_a/m59"})
    ids = {sp["items"][i].context_id for c in CLASSES for g in sp["order"][c] for i in g}
    ids |= {sp["items"][i].context_id for c in CLASSES for i in sp["eval"][c]}
    assert "tg_spam_xtg/tg_spam_a/m59" not in ids and "tg_spam_xtg/tg_spam_a/m59e" not in ids


def test_a_pool_takes_n_bindings_per_class_and_smaller_n_is_a_prefix():
    sp = ladder_split(_items(), CLASSES, eval_n=20)
    small, st = ladder_pool(sp, {"base": 50, "compute": 30, "tg": 10})
    big, _ = ladder_pool(sp, {"base": 100, "compute": 60, "tg": 20})
    fam = collections.Counter(family_of(it) for it in small)
    assert 50 <= fam["base"] <= 51 and fam["xrq"] == 30 and fam["xtg"] == 10
    key = lambda it: (it.context_id, it.question_id)  # noqa: E731
    assert {key(i) for i in small} <= {key(i) for i in big}
    assert st["per_class"]["compute"] == {"n": 30, "bindings": 30, "exhausted": False}


def test_a_class_asked_for_more_than_it_has_is_used_whole_and_marked_exhausted():
    sp = ladder_split(_items(), CLASSES, eval_n=20)
    pool, st = ladder_pool(sp, {"base": 10, "compute": 10_000, "tg": 10})
    assert st["per_class"]["compute"]["bindings"] == 80 and st["per_class"]["compute"]["exhausted"] is True


def test_edit_links_stay_inside_the_pool():
    sp = ladder_split(_items(), CLASSES, eval_n=5)
    pool, _ = ladder_pool(sp, {"base": 10, "compute": 10, "tg": 1000})
    for i, it in enumerate(pool):
        for j in (it.pair, it.contrast):
            assert j is None or 0 <= j < len(pool)
    assert any(it.contrast is not None for it in pool)
