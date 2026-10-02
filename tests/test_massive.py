"""decidophobia.data.massive 的测试: 只做评估的留出数据集, 训练里一条都不出现.

跑:  PYTHONPATH=src .venv/bin/python tests/test_massive.py
data/massive/amazon-massive-dataset-1.0.tar.gz 不在盘上时第一次用到会自动下载 (39.5MB).
"""

import json
import pathlib
import random
import shutil
import tempfile
import urllib.request

from _runner import run
from decidophobia.data.paths import asset_path
from decidophobia.data.massive import DEFAULT_DIR, TARBALL, URL, load_massive


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


def test_desc_names_are_key_and_description_with_the_same_class_ids():
    """labels="desc": 同一个类 id 显示「原始 intent 名: description」(description 取自 datasets/label-descriptions/massive.json),
    与推理服务、synth-v5 的选项同一种写法; train 分区同样可用."""
    raw, te = load_massive(), load_massive(labels="desc")
    desc = json.loads(asset_path("label-descriptions/massive.json").read_text())
    assert te.names == {c: f"{n}: {desc[n]}" for c, n in raw.names.items()}
    assert te.queries == raw.queries and te.labels == raw.labels
    assert load_massive(partition="train", labels="desc").names == te.names


def test_full_menu_lists_every_intent_exactly_once():
    te = load_massive()
    exs = te.build_examples(list(range(60)), (60, 60), random.Random(0))
    assert len(exs) == 2974
    assert all(sorted(ex.options) == list(range(60)) for ex in exs[:50])
    assert all(ex.options[ex.gold_idx] == ex.label for ex in exs[:50])


def test_a_missing_tarball_is_downloaded_from_amazon():
    """空目录里第一次读: 从 Amazon 的 S3 下 tarball (换成拷贝盘上那份), 解出 en-US.jsonl, 读出与默认目录相同的数据."""
    urls = []

    def fake(url, dest):
        urls.append(url)
        shutil.copy(DEFAULT_DIR / TARBALL, dest)

    with tempfile.TemporaryDirectory() as d:
        real, urllib.request.urlretrieve = urllib.request.urlretrieve, fake
        try:
            te = load_massive(pathlib.Path(d) / "massive")
        finally:
            urllib.request.urlretrieve = real
        assert (pathlib.Path(d) / "massive" / TARBALL).exists()
    assert urls == [URL] and URL.startswith("https://amazon-massive-nlu-dataset.s3.amazonaws.com/")
    assert te.queries == load_massive().queries


if __name__ == "__main__":
    run(globals())
