"""decidophobia.data.telegram 的测试: arashdn/telegram-research 的 v2 (Telegram 频道的走红消息, 波斯语, 带类别与情感标注).

跑:  PYTHONPATH=src .venv/bin/python tests/test_telegram.py
读语料的那几项要 data/telegram/tg_v2_public.zip 在盘上 (不自动下载, md5 钉死).
"""

from _runner import run
from decidophobia.data.telegram import cache_path, load_telegram_viral, merge_copies, post_text


# --------------------------------------------------------------------------
# 一条帖子 -> 一段文本
# --------------------------------------------------------------------------


def test_post_text_turns_the_crawlers_literal_backslash_n_into_newlines():
    r"""爬虫把换行存成了两个字符 \n, 库里原样留着."""
    assert post_text("line one\\nline two\\n\\n\\n\\nline three", "") == "line one\nline two\n\nline three"


def test_post_text_uses_the_caption_when_the_message_is_empty():
    """带图或视频的帖子, 文字在 caption 里; 两个字段不会同时有字."""
    assert post_text("", "caption here") == "caption here"
    assert post_text(None, "caption here") == "caption here"


def test_post_text_trims_spaces_and_drops_control_characters():
    assert post_text("  a\x07b   c  \\n  d ", None) == "ab c\nd"


def test_post_text_is_cut_to_max_body_characters():
    assert len(post_text("word " * 1000, None, max_body=100)) <= 100


def test_post_text_of_an_empty_post_is_empty():
    assert post_text("", None) == ""


# --------------------------------------------------------------------------
# 同一段文本的几份拷贝并成一条
# --------------------------------------------------------------------------


def test_merge_copies_counts_each_copys_labels_and_keeps_first_seen_order():
    """同一段文本常被好几个频道转发, 每份拷贝各标各的; 并成一条, 标签记成每个值出现几次."""
    posts = [
        {"id": 5, "text": "b", "tag": 21, "sentiment": 0},
        {"id": 3, "text": "a", "tag": 4, "sentiment": -1},
        {"id": 9, "text": "b", "tag": 19, "sentiment": 0},
        {"id": 7, "text": "b", "tag": 21, "sentiment": 1},
    ]
    assert merge_copies(posts) == [
        {"text": "b", "ids": [5, 9, 7], "tags": {21: 2, 19: 1}, "sentiments": {0: 2, 1: 1}},
        {"text": "a", "ids": [3], "tags": {4: 1}, "sentiments": {-1: 1}},
    ]


def test_merge_copies_leaves_out_sentiments_outside_minus_one_to_one():
    """sentiment 只有 -1 / 0 / 1 三档; 库里各有一条 2 与 -2, 那份拷贝的情感不计 (类别照计)."""
    posts = [{"id": 1, "text": "a", "tag": 20, "sentiment": 2}, {"id": 2, "text": "a", "tag": 20, "sentiment": 1}]
    assert merge_copies(posts) == [{"text": "a", "ids": [1, 2], "tags": {20: 2}, "sentiments": {1: 1}}]


def test_merge_copies_drops_empty_texts():
    assert merge_copies([{"id": 1, "text": "", "tag": 20, "sentiment": 0}]) == []


# --------------------------------------------------------------------------
# 语料 (要原始包在盘上)
# --------------------------------------------------------------------------


def test_the_corpus_is_the_18566_viral_messages_merged_by_text():
    c = load_telegram_viral()
    assert sum(sum(p["tags"].values()) for p in c.posts) == 18566 - 158, "158 条既无 message 也无 caption"
    assert len({p["text"] for p in c.posts}) == len(c.posts) and 15000 < len(c.posts) < 15400
    assert all(p["text"] and p["tags"] and set(p["sentiments"]) <= {-1, 0, 1} for p in c.posts)


def test_the_24_tags_hang_under_8_super_tags_as_in_the_dump():
    c = load_telegram_viral()
    assert sorted(c.tags) == list(range(1, 25)) and sorted(c.super_tags) == list(range(1, 9))
    assert c.tags[19] == ("Promotion/offer", 7) and c.tags[21] == ("تبلیغات spam", 7)
    assert c.super_tags[7] == "promotion/spam"
    assert {t for p in c.posts for t in p["tags"]} == set(range(1, 25))


def test_a_known_post_keeps_its_text_and_labels():
    c = load_telegram_viral()
    p = next(p for p in c.posts if 15140 in p["ids"])
    assert p["ids"] == [15140, 126214], "这条键盘广告被转发过一次, 两份拷贝都标成广告"
    assert p["text"].startswith("☂ڪیبوردِ زیــــبانِویس☂\n") and p["tags"] == {21: 2} and p["sentiments"] == {0: 2}


def test_the_parsed_corpus_is_cached_and_reread_identically():
    a = load_telegram_viral()
    assert cache_path().exists()
    assert load_telegram_viral() == a


def test_a_dump_with_the_wrong_md5_is_refused():
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        (pathlib.Path(d) / "telegram").mkdir()
        (pathlib.Path(d) / "telegram" / "tg_v2_public.zip").write_bytes(b"not the zip")
        try:
            load_telegram_viral(data_dir=d)
        except RuntimeError as e:
            assert "md5" in str(e)
        else:
            raise AssertionError("wrong md5 was accepted")


if __name__ == "__main__":
    run(globals())
