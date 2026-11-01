"""Translated materials (the data's i18n/<domain>.json): each translation is another variant of its material,
and an example drawn on a translated variant shows the question and options in that language.

Run:  PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/test_synth_v5_i18n.py
"""

import json
import random
import shutil

from sors.data.synth_v5 import item_example, load_synth_v5, pair_examples
from test_synth_v5 import _data

ZH_TEXT = "我想退网，怎么注销我的号码"
OVERLAY = {
    "domain": "telecom",
    "contexts": {"telecom_cancel_contract#0": {"zh": ZH_TEXT}},
    "questions": {
        "intent": {"zh": {"ask": ["哪个选项最符合这条消息？"],
                          "options": {"cancel_contract": "彻底解除合约", "port_number_out": "把号码转走要用的代码",
                                      "view_latest_bill": "最新账单及各项费用明细", "3g_switch_off": None}}},
        "direct": {"zh": {"ask": ["客户有没有直接说出想要什么？", "请求是否被直接说出？"],
                          "options": {"no": "只描述了处境", "yes": "直接说出了请求"}}},
        "detail_cancel_contract": {"zh": {"ask": ["客户有没有问到费用？"], "options": {"no": None, "yes": None}}},
    },
}


def _items():
    d = _data()
    (d / "i18n").mkdir()
    (d / "i18n" / "telecom.json").write_text(json.dumps(OVERLAY, ensure_ascii=False), encoding="utf-8")
    try:
        return load_synth_v5(d)
    finally:
        shutil.rmtree(d)


def _bound(items, cid, qid):
    return next(it for it in items if it.context_id == cid and it.question_id == qid)


def test_a_translation_is_one_more_variant_tagged_with_its_language():
    it = _bound(_items(), "telecom_cancel_contract#0", "direct")
    assert it.texts == ["i want out, how do i close my line", ZH_TEXT]
    assert it.langs == ("en", "zh")


def test_a_translated_variant_shows_the_question_and_the_options_in_its_language():
    it = _bound(_items(), "telecom_cancel_contract#0", "direct")
    seen = {}
    rng = random.Random(0)
    for _ in range(60):
        ex = item_example(it, rng, 0.0, fallback_rate=0.0)
        seen[ex.query] = ex
        if ex.query == ZH_TEXT:
            assert ex.question in OVERLAY["questions"]["direct"]["zh"]["ask"]
            assert sorted(ex.option_names) == ["no: 只描述了处境", "yes: 直接说出了请求"]
            assert ex.option_names[ex.gold_idx] == "yes: 直接说出了请求"
        else:
            assert ex.question in it.asks and "yes: The request is named" in ex.option_names
    assert set(seen) == set(it.texts)


def test_a_masked_translated_example_shows_only_the_untranslated_keys():
    it = _bound(_items(), "telecom_cancel_contract#0", "direct")
    rng = random.Random(1)
    exs = [item_example(it, rng, 1.0, fallback_rate=0.0) for _ in range(40)]
    assert all(sorted(e.option_names) == ["no", "yes"] for e in exs)


def test_the_fallback_option_follows_the_language_of_the_variant():
    import dataclasses
    it = _bound(_items(), "telecom_cancel_contract#0", "intent")
    it = dataclasses.replace(it, fallback_row="none_of_these: None of these options applies", fallback_bare="none_of_these",
                             i18n={**it.i18n, "zh": {**it.i18n["zh"], "fallback_row": "none_of_these: 这些选项都不适用"}})
    rng = random.Random(2)
    rows = {e.query: e.option_names for e in (item_example(it, rng, 0.0, fallback_rate=1.0) for _ in range(40))}
    assert "none_of_these: 这些选项都不适用" in rows[ZH_TEXT]
    assert "none_of_these: None of these options applies" in rows[it.texts[0]]


def test_untranslated_materials_draw_exactly_as_before():
    plain = _data()
    try:
        before = load_synth_v5(plain)
    finally:
        shutil.rmtree(plain)
    after = _items()
    for a, b in zip(before, after):
        if b.context_id == "telecom_cancel_contract#0":
            continue
        ra, rb = random.Random(5), random.Random(5)
        ea, eb = item_example(a, ra, 0.5), item_example(b, rb, 0.5)
        assert (ea.query, ea.question, ea.option_names, ea.gold_idx) == (eb.query, eb.question, eb.option_names, eb.gold_idx)


def test_both_halves_of_a_pair_share_the_language_of_the_drawn_variant():
    items = _items()
    a = _bound(items, "telecom_cancel_contract#0", "direct")
    rng = random.Random(3)
    for _ in range(30):
        x, y = pair_examples(a, a, rng, 0.0)
        assert x.query == y.query and x.question == y.question and sorted(x.option_names) == sorted(y.option_names)
