"""decidophobia.massive 的测试: 只做评估的留出数据集, 训练里一条都不出现.

跑:  PYTHONPATH=src .venv/bin/python tests/test_massive.py
需要 data/massive/amazon-massive-dataset-1.0.tar.gz 在盘上 (39.5MB, 不自动下载).
"""

import json
import random

from _runner import run
from decidophobia.label_names import DESC_DIR
from decidophobia.massive import load_massive


def test_test_split_has_2974_utterances_over_60_intents():
    te = load_massive()
    assert len(te.queries) == 2974
    assert len(te.names) == 60
    assert set(te.labels) == set(range(60)) - {10}, "cooking_query (id 10) 在 test 里 0 条、train 里 4 条; 其余 59 类都有"
    assert te.context_label != "Customer message", "上下文标签要换成这个数据集自己的, 别把 Banking77 的框架带过来"
    assert te.qtype == "choice"


def test_intent_ids_follow_sorted_raw_names_and_the_names_are_shown_raw():
    """类 id 按原始 intent 名字母序, 与 banking77 同一约定; 菜单上显示的就是原始名, 下划线和粘连的复合词都不动."""
    te = load_massive()
    assert te.names[0] == "alarm_query"
    assert te.names[59] == "weather_query"
    assert "iot_hue_lightchange" in te.names.values()


def test_desc_names_are_the_descriptions_with_the_same_class_ids():
    """labels="desc": 同一个类 id 显示 datasets/label-descriptions/massive.json 里它的 description; train 分区同样可用."""
    raw, te = load_massive(), load_massive(labels="desc")
    desc = json.loads((DESC_DIR / "massive.json").read_text())
    assert te.names == {c: desc[n] for c, n in raw.names.items()}
    assert te.queries == raw.queries and te.labels == raw.labels
    assert load_massive(partition="train", labels="desc").names == te.names


def test_full_menu_lists_every_intent_exactly_once():
    te = load_massive()
    exs = te.build_examples(list(range(60)), (60, 60), random.Random(0))
    assert len(exs) == 2974
    assert all(sorted(ex.options) == list(range(60)) for ex in exs[:50])
    assert all(ex.options[ex.gold_idx] == ex.label for ex in exs[:50])


if __name__ == "__main__":
    run(globals())
