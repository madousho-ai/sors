"""sors.data.synth_v5 的测试: v5 的材料与绑定读成逐题的样本, 以及训练时按目标配比抽题.

数据用临时目录现造 (复制仓库里的 schema.py, 再写两个小领域). 真实数据的读取见 tests/test_train_cli.py.

跑:  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python tests/test_synth_v5.py
"""

import collections
import json
import pathlib
import random
import shutil
import tempfile

from _runner import run
from sors.core.menu import row_alignment, with_partners
from sors.data.paths import asset_path
from sors.data.synth_v5 import (V5Rounds, V5Sampler, item_example, load_synth_v5, mix_at, parse_mix, pair_examples,
                                        parse_passes)

REPO = pathlib.Path(__file__).resolve().parent.parent
INTENTS = {"cancel_contract": "Cancelling the contract entirely", "port_number_out": "Code for taking the number away",
           "view_latest_bill": "Latest bill and a breakdown of the charges", "3g_switch_off": None}

TELECOM_Q = [
    {"id": "intent", "ask": ["Which option best describes the message?"], "options": INTENTS},
    {"id": "direct", "ask": ["Does the customer say outright what they want?", "Is the request named outright?"],
     "options": {"no": "Only the situation is described", "yes": "The request is named"}},
    {"id": "detail_cancel_contract", "kind": "detail", "ask": ["does the customer ask about fees?"],
     "options": {"no": None, "yes": None}},
    {"id": "detail_view_latest_bill", "kind": "detail", "ask": ["does the customer give the amount?"],
     "options": {"no": None, "yes": None}},
    {"id": "pickup", "ask": ["Which alert should be taken next?"], "options": ["LOG-0719", "NET-0457", "EDR-2310"]},
    {"id": "level", "type": "score", "ask": ["How severe is it?"], "options": ["Cosmetic", "Degraded", "Down"]},
]
TELECOM_C = [
    {"id": "telecom_cancel_contract#0", "label": "Customer message", "style": "short", "goal": "breadth",
     "text": "i want out, how do i close my line",
     "questions": [{"question": "intent", "answer": "cancel_contract", "maskable": True, "goal": "long_menu"},
                   {"question": "direct", "answer": "yes", "maskable": True},
                   {"question": "detail_cancel_contract", "answer": "no"}]},
    {"id": "telecom_view_latest_bill#0", "label": "Customer message", "style": "short", "goal": "breadth",
     "text": "whats on my bill this month",
     "questions": [{"question": "intent", "answer": {"view_latest_bill": 3, "cancel_contract": 1}, "goal": "long_menu"},
                   {"question": "detail_view_latest_bill", "answer": "no"}]},
    {"id": "telecom_queue_a", "label": "Security on-call screen", "goal": "breadth",
     "text": ["LOG-0719 raised 12:47, no owner. NET-0457 raised 13:58, no owner.",
              "Two alerts have no owner: NET-0457 (raised 13:58) and LOG-0719 (raised 12:47).",
              "Unowned: LOG-0719 at 12:47 and NET-0457 at 13:58."],
     "questions": [{"question": "pickup", "answer": "LOG-0719"}]},
    {"id": "telecom_rule_1", "label": "", "goal": "complex",
     "text": {"rule": "Three failed checks mean the line is down.", "log": "3 checks failed"},
     "questions": [{"question": "level", "answer": {"1": 1, "2": 3}}]},
]
HOTEL_Q = [{"id": "intent", "ask": ["Which option best describes the message?"],
            "options": {"late_checkout": "Leaving the room later than usual", "extra_towels": "More towels in the room"}}]
HOTEL_C = [{"id": f"hotel_{k}#{i}", "label": "Customer message", "goal": "long_menu", "text": t,
            "questions": [{"question": "intent", "answer": k}]}
           for i, (k, t) in enumerate([("late_checkout", "can we stay till 2"), ("extra_towels", "need more towels pls"),
                                       ("late_checkout", "flight is at 6, can we keep the room longer")])]


def _data(**over):
    d = pathlib.Path(tempfile.mkdtemp())
    shutil.copy(asset_path("synth-intents-v5.3/schema.py"), d)
    files = {"telecom": (TELECOM_Q, TELECOM_C), "hotel": (HOTEL_Q, HOTEL_C), **over}
    for dom, (qs, cs) in files.items():
        (d / f"{dom}.questions.json").write_text(json.dumps({"domain": dom, "questions": qs}))
        (d / f"{dom}.contexts.json").write_text(json.dumps({"domain": dom, "contexts": cs}))
    return d


def _items():
    return load_synth_v5(_data())


def _find(items, cid, qid):
    return next(it for it in items if it.context_id == cid and it.question_id == qid)


# ---- 读取

def test_each_binding_is_one_question_and_an_unbound_question_is_not_asked():
    items = _items()
    assert len(items) == 3 + 2 + 1 + 1 + 3
    assert {it.question_id for it in items if it.context_id == "telecom_view_latest_bill#0"} == \
        {"intent", "detail_view_latest_bill"}
    assert not any(it.question_id == "direct" and it.context_id != "telecom_cancel_contract#0" for it in items)


def test_keyed_options_show_key_and_description_and_a_null_description_shows_the_key():
    it = _find(_items(), "telecom_cancel_contract#0", "intent")
    assert it.rows == ["cancel_contract: Cancelling the contract entirely",
                       "port_number_out: Code for taking the number away",
                       "view_latest_bill: Latest bill and a breakdown of the charges", "3g_switch_off"], it.rows
    assert it.bare == ["cancel_contract", "port_number_out", "view_latest_bill", "3g_switch_off"]
    assert it.qtype == "choice" and it.label == "Customer message"


def test_yes_no_is_a_bool_question_shown_as_no_then_yes_with_descriptions():
    it = _find(_items(), "telecom_cancel_contract#0", "direct")
    assert it.qtype == "bool" and it.rows == ["no: Only the situation is described", "yes: The request is named"]
    assert it.gold == 1 and it.target is None
    bare = _find(_items(), "telecom_cancel_contract#0", "detail_cancel_contract")
    assert bare.rows == ["no", "yes"] and bare.bare is None, "说明全是 null 的题没有可遮的东西"


def test_listed_options_and_score_levels_are_shown_as_written_and_cannot_be_masked():
    items = _items()
    p = _find(items, "telecom_queue_a", "pickup")
    assert p.rows == ["LOG-0719", "NET-0457", "EDR-2310"] and p.gold == 0 and p.bare is None
    s = _find(items, "telecom_rule_1", "level")
    assert s.rows == ["Cosmetic", "Degraded", "Down"] and s.qtype == "choice" and s.bare is None


def test_a_distribution_is_normalised_and_its_most_likely_option_is_the_gold():
    it = _find(_items(), "telecom_view_latest_bill#0", "intent")
    assert it.target == [0.25, 0.0, 0.75, 0.0] and it.gold == 2
    s = _find(_items(), "telecom_rule_1", "level")
    assert s.target == [0.0, 0.25, 0.75] and s.gold == 2


def test_a_binding_goal_overrides_the_contexts_goal():
    items = _items()
    assert _find(items, "telecom_cancel_contract#0", "intent").goal == "long_menu"
    assert _find(items, "telecom_cancel_contract#0", "direct").goal == "breadth"


def test_kind_defaults_to_the_question_id():
    items = _items()
    assert _find(items, "telecom_cancel_contract#0", "detail_cancel_contract").kind == "detail"
    assert _find(items, "telecom_cancel_contract#0", "direct").kind == "direct"


def test_json_text_is_written_as_indented_json():
    it = _find(_items(), "telecom_rule_1", "level")
    assert it.texts == [json.dumps(TELECOM_C[3]["text"], indent=2, ensure_ascii=False)]


def test_data_that_fails_its_format_check_is_refused():
    bad = [dict(TELECOM_C[0], questions=[{"question": "intent", "answer": "top_up"}]), *TELECOM_C[1:]]
    try:
        load_synth_v5(_data(telecom=(TELECOM_Q, bad)))
    except ValueError as e:
        assert "telecom" in str(e)
        return
    raise AssertionError("loaded a binding whose answer is not an option")


# ---- 一道题出成样本

def test_an_example_shuffles_the_rows_and_the_target_and_gold_follow():
    it = _find(_items(), "telecom_view_latest_bill#0", "intent")
    rng = random.Random(0)
    orders = set()
    for _ in range(40):
        ex = item_example(it, rng, mask_rate=0.0)
        orders.add(tuple(ex.options))
        assert dict(zip(ex.options, ex.target)) == dict(enumerate(it.target))
        assert ex.options[ex.gold_idx] == 2 and ex.option_names[ex.gold_idx].startswith("view_latest_bill:")
    assert len(orders) > 10


def test_an_example_draws_one_of_the_questions_wordings():
    it = _find(_items(), "telecom_cancel_contract#0", "direct")
    rng = random.Random(0)
    assert {item_example(it, rng, 0.0).question for _ in range(40)} == set(it.asks)


def test_a_material_with_several_phrasings_gives_its_pair_copy_a_different_phrasing():
    it = _find(_items(), "telecom_queue_a", "pickup")
    rng = random.Random(0)
    seen = set()
    for _ in range(60):
        ex = item_example(it, rng, 0.0)
        assert ex.query in it.texts and ex.partner_query in it.texts and ex.query != ex.partner_query
        seen.add(ex.query)
        a, b = with_partners([ex], rng)
        assert b.query == ex.partner_query and [b.options[j] for j in row_alignment(a, b)] == a.options
    assert seen == set(it.texts)


def test_a_material_with_one_phrasing_pairs_with_itself():
    ex = item_example(_find(_items(), "telecom_cancel_contract#0", "direct"), random.Random(0), 0.0)
    assert ex.partner_query is None


def test_masking_hides_every_description_of_a_maskable_binding_and_the_pair_copy_hides_them_too():
    items = _items()
    rng = random.Random(0)
    it = _find(items, "telecom_cancel_contract#0", "intent")
    for _ in range(20):
        ex = item_example(it, rng, mask_rate=1.0)
        assert sorted(ex.option_names) == sorted(it.bare)
        a, b = with_partners([ex], rng)
        assert sorted(b.option_names) == sorted(it.bare)
    kept = _find(items, "telecom_view_latest_bill#0", "intent")  # 这个绑定没写 maskable
    assert sorted(item_example(kept, rng, 1.0).option_names) == sorted(kept.rows)
    assert all(sorted(item_example(it, rng, 0.0).option_names) == sorted(it.rows) for _ in range(20))


def test_masking_at_rate_one_half_hides_about_half_the_time():
    it = _find(_items(), "telecom_cancel_contract#0", "direct")
    rng = random.Random(0)
    hidden = sum(item_example(it, rng, 0.5).option_names[0] in ("no", "yes") for _ in range(2000))
    assert 900 < hidden < 1100, hidden


# ---- 目标配比

GOALS = {"long_menu", "breadth", "complex"}


def test_no_mix_gives_every_goal_in_the_data_the_same_share():
    assert parse_mix(None, GOALS) == [(0, {"long_menu": 1.0, "breadth": 1.0, "complex": 1.0})]


def test_a_mix_without_steps_holds_for_the_whole_run():
    assert parse_mix("long_menu=4,breadth=2,complex=1", GOALS) == [(0, {"long_menu": 4.0, "breadth": 2.0, "complex": 1.0})]


def test_a_staged_mix_switches_at_the_given_steps():
    stages = parse_mix("0: long_menu=6,breadth=2,complex=1; 1000: long_menu=3,breadth=2,complex=2", GOALS)
    assert mix_at(stages, 1) == mix_at(stages, 999) == {"long_menu": 6.0, "breadth": 2.0, "complex": 1.0}
    assert mix_at(stages, 1000) == mix_at(stages, 5000) == {"long_menu": 3.0, "breadth": 2.0, "complex": 2.0}


def test_every_goal_in_the_data_keeps_a_share_in_every_stage():
    for spec in ("long_menu=4,breadth=2", "long_menu=4,breadth=2,complex=0",
                 "0: long_menu=1,breadth=1,complex=1; 500: long_menu=1,breadth=1"):
        try:
            parse_mix(spec, GOALS)
        except ValueError as e:
            assert "complex" in str(e), e
            continue
        raise AssertionError(f"{spec!r} dropped a goal")


def test_a_goal_without_data_or_unknown_is_refused():
    for spec in ("long_menu=1,breadth=1,complex=1,long_context=1", "long_menu=1,breadth=1,complex=1,fun=1"):
        try:
            parse_mix(spec, GOALS)
        except ValueError:
            continue
        raise AssertionError(f"{spec!r} accepted")


def test_a_staged_mix_starts_at_step_zero():
    try:
        parse_mix("100: long_menu=1,breadth=1,complex=1", GOALS)
    except ValueError:
        return
    raise AssertionError("a mix with nothing before step 100 accepted")


# ---- 抽题

def _which(items, ex):
    """抽出来的样本是哪一个绑定: 读的是它的某种说法, 菜单是它的选项 (没遮说明)."""
    hits = [it for it in items if ex.query in it.texts and sorted(ex.option_names) == sorted(it.rows)
            and ex.question in it.asks]
    assert len(hits) == 1, (ex, hits)
    return hits[0]


def test_goals_are_drawn_by_the_mix_and_the_mix_follows_the_step():
    items = _items()
    s = V5Sampler(items, parse_mix("0: long_menu=6,breadth=3,complex=1; 50: long_menu=1,breadth=1,complex=8", GOALS), 0.0)
    rng = random.Random(0)
    first, later = collections.Counter(), collections.Counter()
    for step in range(1, 101):
        for ex in s(40, rng):
            (first if step < 50 else later)[_which(items, ex).goal] += 1
    n1, n2 = sum(first.values()), sum(later.values())
    assert abs(first["long_menu"] / n1 - 0.6) < 0.03 and abs(first["complex"] / n1 - 0.1) < 0.03, first
    assert abs(later["complex"] / n2 - 0.8) < 0.03, later


def test_within_a_goal_domains_are_equal_whatever_their_size():
    """long_menu 有 telecom 两份材料、hotel 三份: 仍是各领域一半. breadth 只有 telecom."""
    items = _items()
    s = V5Sampler(items, parse_mix("long_menu=1,breadth=1,complex=1", GOALS), 0.0)
    rng = random.Random(1)
    domains = collections.Counter(_which(items, ex).domain for _ in range(300) for ex in s(10, rng)
                                  if _which(items, ex).goal == "long_menu")
    assert abs(domains["hotel"] / sum(domains.values()) - 0.5) < 0.04, domains


def test_within_a_domain_kinds_are_equal_however_many_contexts_each_has():
    """telecom 的 breadth 有三种题型: direct (1 份材料)、detail (2 份)、pickup (1 份), 各占三分之一."""
    items = _items()
    s = V5Sampler(items, parse_mix("long_menu=1,breadth=1,complex=1", GOALS), 0.0)
    rng = random.Random(2)
    kinds = collections.Counter(_which(items, ex).kind for _ in range(600) for ex in s(6, rng)
                                if _which(items, ex).goal == "breadth")
    n = sum(kinds.values())
    assert set(kinds) == {"direct", "detail", "pickup"} and all(abs(v / n - 1 / 3) < 0.03 for v in kinds.values()), kinds


def test_every_binding_is_drawn():
    items = _items()
    s = V5Sampler(items, parse_mix(None, GOALS), 0.0)
    rng = random.Random(3)
    seen = {(it.context_id, it.question_id) for _ in range(300) for it in (_which(items, ex) for ex in s(8, rng))}
    assert seen == {(it.context_id, it.question_id) for it in items}


# ---- 成对: other 版与原题一起抽出来

PICKUP_OTHER = {"id": "pickup_other", "kind": "pickup", "other_of": "pickup", "ask": ["Which alert should be taken next?"],
                "options": ["LOG-0719", "NET-0457", "EDR-2310", "None of these alerts"]}
INTENT_OTHER = {"id": "intent_other", "kind": "intent", "other_of": "intent",
                "ask": ["Which option best describes the message?"], "options": {**INTENTS, "other": "None of these"}}
NOTHING_FITS = {"id": "telecom_amb/queue/0", "label": "Security on-call screen", "goal": "ambiguous",
                "text": ["The alert queue is empty; someone asks why the floor 2 printer is jammed.",
                         "No alerts are waiting. The only question today: why is the printer on floor 2 jammed?"],
                "questions": [{"question": "pickup", "answer": {"LOG-0719": 1, "NET-0457": 1, "EDR-2310": 1}},
                              {"question": "pickup_other", "answer": "None of these alerts"}]}
PAIR_GOALS = GOALS | {"ambiguous"}


def _paired():
    control = dict(TELECOM_C[0], questions=TELECOM_C[0]["questions"] + [
        {"question": "intent_other", "answer": "cancel_contract", "maskable": True, "goal": "long_menu"}])
    return load_synth_v5(_data(telecom=(TELECOM_Q + [PICKUP_OTHER, INTENT_OTHER],
                                        [control, *TELECOM_C[1:], NOTHING_FITS])))


def test_the_two_halves_of_a_pair_point_at_each_other_and_other_items_have_no_pair():
    items = _paired()
    for cid, q in (("telecom_amb/queue/0", "pickup"), ("telecom_cancel_contract#0", "intent")):
        a, b = _find(items, cid, q), _find(items, cid, f"{q}_other")
        assert items[a.pair] == b and items[b.pair] == a
    assert _find(items, "telecom_queue_a", "pickup").pair is None
    assert sum(it.pair is not None for it in items) == 4


def _draws(items, mix, n, steps, mask=0.0, seed=0):
    s = V5Sampler(items, parse_mix(mix, PAIR_GOALS), mask)
    rng = random.Random(seed)
    return [s(n, rng) for _ in range(steps)]


def _halves(batch):
    """批里成对的两半 (other 版紧跟在原题后面)."""
    return [(a, b) for a, b in zip(batch, batch[1:]) if len(b.options) == len(a.options) + 1
            and b.query == a.query and b.question == a.question]


def test_diagnostic_pair_gives_both_halves_on_one_phrasing_and_one_wording():
    items = _paired()
    base, other = (_find(items, "telecom_amb/queue/0", q) for q in ("pickup", "pickup_other"))
    rng = random.Random(0)
    queue = [pair_examples(base, other, rng, 0.0) for _ in range(250)]
    assert {a.query for a, _ in queue} == set(NOTHING_FITS["text"])
    for a, b in queue:
        assert sorted(b.option_names) == sorted(a.option_names + ["None of these alerts"])
        assert a.target == [1 / 3] * 3 and b.option_names[b.gold_idx] == "None of these alerts"
    assert len({b.option_names.index("None of these alerts") for _, b in queue}) == 4, "other 那一行的位置跟着打乱"


def test_a_pair_counts_as_one_draw():
    items = _paired()
    for batch in _draws(items, "long_menu=1,breadth=1,complex=1,ambiguous=3", 10, 50):
        assert len(batch) == 10, len(batch)


def test_both_halves_of_a_diagnostic_pair_are_masked_together():
    items = _paired()
    both = collections.Counter()
    base, other = (_find(items, "telecom_cancel_contract#0", q) for q in ("intent", "intent_other"))
    rng = random.Random(0)
    for _ in range(200):
        a, b = pair_examples(base, other, rng, 0.5)
        both[(any(": " in n for n in a.option_names), any(": " in n for n in b.option_names))] += 1
    assert set(both) == {(True, True), (False, False)}, both


# ---- 按轮抽: 一轮把每个绑定出一遍, 列出的目标出几遍

def test_passes_default_every_goal_to_one_and_take_the_listed_counts():
    assert parse_passes(None, GOALS) == {"long_menu": 1, "breadth": 1, "complex": 1}
    assert parse_passes("complex=3", GOALS) == {"long_menu": 1, "breadth": 1, "complex": 3}
    assert parse_passes("complex=3,breadth=2", GOALS) == {"long_menu": 1, "breadth": 2, "complex": 3}


def test_passes_refuse_an_unknown_goal_or_a_count_that_is_not_a_positive_whole_number():
    for spec in ("fun=2", "long_context=2", "complex=0", "complex=1.5", "complex"):
        try:
            parse_passes(spec, GOALS)
        except ValueError:
            continue
        raise AssertionError(f"{spec!r} accepted")


def _round_order(items, passes, n, draws, seed=0):
    """按轮抽 draws 批, 每批 n 次抽取; 返回抽出来的绑定, 按出场顺序."""
    s = V5Rounds(items, parse_passes(passes, GOALS), 0.0)
    rng = random.Random(seed)
    return s, [(_which(items, ex).context_id, _which(items, ex).question_id) for _ in range(draws) for ex in s(n, rng)]


def test_a_round_draws_every_binding_once_and_a_listed_goal_as_many_times_as_its_passes():
    """10 个绑定, complex 只有 telecom_rule_1 一个, 出 3 遍: 一轮 12 次抽取."""
    items = _items()
    s, got = _round_order(items, "complex=3", 4, 3)
    assert s.round_size == 12
    want = collections.Counter({(it.context_id, it.question_id): 3 if it.goal == "complex" else 1 for it in items})
    assert collections.Counter(got) == want


def test_the_next_round_starts_where_one_runs_out_in_a_new_order():
    """一批跨过两轮的边界也照样取满; 每一轮各是一遍完整的数据, 两轮的顺序不同."""
    items = _items()
    _, got = _round_order(items, None, 7, 6)  # 42 次 = 4 轮 (每轮 10) 再加 2
    rounds = [got[i:i + 10] for i in range(0, 40, 10)]
    whole = collections.Counter((it.context_id, it.question_id) for it in items)
    assert all(collections.Counter(r) == whole for r in rounds)
    assert len({tuple(r) for r in rounds}) == 4


def test_a_pair_is_one_draw_of_a_round_and_brings_one_random_view():
    items = _paired()
    s = V5Rounds(items, parse_passes(None, PAIR_GOALS), 0.0)
    assert s.round_size == len(items) - 2  # 两对, 各少算一次
    rng = random.Random(0)
    batches = [s(s.round_size, rng) for _ in range(4)]
    assert all(len(b) == s.round_size for b in batches)
    got = collections.Counter((_which(items, ex).context_id, _which(items, ex).question_id.removesuffix("_other"))
                              for b in batches for ex in b)
    assert set(got.values()) == {4}, got


if __name__ == "__main__":
    run(globals())
