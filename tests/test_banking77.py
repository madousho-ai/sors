"""decidophobia.banking77 的测试: 菜单上显示原始 label 名或 description.

跑:  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python tests/test_banking77.py
需要 data/banking77/{train,test}.csv (第一次跑会从上游钉死的 commit 下载).
"""

import json

from _runner import run
from decidophobia.banking77 import load_banking77
from decidophobia.label_names import DESC_DIR


def test_names_are_the_raw_labels_unchanged():
    """下划线、大小写、问号原样保留; 类 id 按原始名字母序 (大写开头的 Refund_ 排在最前)."""
    tr, te = load_banking77()
    assert len(te.names) == 77 and te.names == tr.names
    assert te.names[0] == "Refund_not_showing_up"
    assert "reverted_card_payment?" in te.names.values()
    assert list(te.names.values()) == sorted(te.names.values())


def test_desc_names_are_the_descriptions_with_the_same_class_ids():
    """labels="desc": 同一个类 id 显示 datasets/label-descriptions/banking77.json 里它的 description; 题目与标签不变."""
    raw_tr, raw_te = load_banking77()
    tr, te = load_banking77(labels="desc")
    desc = json.loads((DESC_DIR / "banking77.json").read_text())
    assert te.names == {c: desc[n] for c, n in raw_te.names.items()}
    assert te.queries == raw_te.queries and te.labels == raw_te.labels and tr.labels == raw_tr.labels


def test_an_unknown_label_style_is_refused():
    try:
        load_banking77(labels="human")
    except ValueError:
        return
    raise AssertionError("labels='human' accepted")


if __name__ == "__main__":
    run(globals())
