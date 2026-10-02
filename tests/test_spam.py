"""sors.data.spam 的测试: spam 语料 (SMS Spam Collection / Enron-Spam / TREC / Telegram 广告标注) 读成 (文本, 是否 spam).

跑:  PYTHONPATH=src .venv/bin/python tests/test_spam.py
读语料的那几项要用 data/sms-spam、data/enron-spam、data/trec-spam、data/telegram 下的原始包; 不在盘上时第一次用到会自动下载 (md5 钉死).
"""

import pathlib
import shutil
import tempfile
import urllib.request

from _runner import run
from sors.data.spam import (DEFAULT_DATA_DIR, MAX_BODY, MAX_GARBLED, SOURCES, URLS, cache_path, dedupe,
                                    email_text, garbled, load_spam)


def _mail(headers: str, body: str | bytes) -> bytes:
    body = body if isinstance(body, bytes) else body.encode()
    return headers.replace("\n", "\r\n").encode() + b"\r\n" + body


# --------------------------------------------------------------------------
# 一封邮件 -> 一段文本
# --------------------------------------------------------------------------


def test_a_plain_email_becomes_its_subject_line_a_blank_line_then_the_body():
    raw = _mail("From: a@b.c\nTo: d@e.f\nSubject: Lunch on Friday\nContent-Type: text/plain\n",
                "Are you free at noon?\nBring the slides.\n")
    assert email_text(raw) == "Subject: Lunch on Friday\n\nAre you free at noon?\nBring the slides."


def test_other_headers_stay_out_of_the_text():
    raw = _mail("Received: from x by y\nMessage-ID: <1@x>\nFrom: a@b.c\nSubject: Hi\n", "Body\n")
    t = email_text(raw)
    assert "Received" not in t and "Message-ID" not in t and "a@b.c" not in t


def test_without_a_subject_header_the_text_is_just_the_body():
    assert email_text(_mail("From: a@b.c\n", "Only the body\n")) == "Only the body"


def test_an_encoded_word_subject_is_decoded():
    raw = _mail("Subject: =?utf-8?b?Q2Fmw6kgbWVudQ==?=\n", "x\n")
    assert email_text(raw).startswith("Subject: Café menu\n\n")


def test_an_html_only_email_keeps_the_visible_text_without_tags_scripts_styles_or_entities():
    html = ("<html><head><style>p {color: red}</style><script>var a = 1;</script></head>"
            "<body><p>Save&nbsp;50% on <b>REFINANCE</b> &amp; more</p><p>Click <a href='http://x'>here</a></p>"
            "<!-- hidden comment --></body></html>")
    raw = _mail("Subject: Offer\nContent-Type: text/html; charset=us-ascii\n", html)
    t = email_text(raw)
    body = t.split("\n\n", 1)[1]
    assert "<" not in body and "color" not in body and "var a" not in body and "hidden comment" not in body
    assert "&amp;" not in body and "&nbsp;" not in body
    assert "Save 50% on REFINANCE & more" in body and "Click here" in body


def test_a_multipart_email_uses_the_plain_part_over_the_html_one():
    raw = _mail("Subject: Both\nMIME-Version: 1.0\nContent-Type: multipart/alternative; boundary=\"XX\"\n",
                "--XX\r\nContent-Type: text/html\r\n\r\n<p>html version</p>\r\n"
                "--XX\r\nContent-Type: text/plain\r\n\r\nplain version\r\n--XX--\r\n")
    assert email_text(raw) == "Subject: Both\n\nplain version"


def test_a_body_in_an_unknown_charset_still_comes_out_as_text():
    raw = _mail("Subject: Odd\nContent-Type: text/plain; charset=DEFAULT\n", "still readable\n")
    assert email_text(raw) == "Subject: Odd\n\nstill readable"


def test_a_body_without_a_charset_is_read_with_the_fallback_encoding():
    """trec06c 的中文邮件常常不声明字符集, 字节是 GBK. 语料给 fallback, 读不通才退回 utf-8."""
    raw = _mail("Subject: x\n", "你好，周五开会".encode("gb18030"))
    assert email_text(raw, fallback="gb18030") == "Subject: x\n\n你好，周五开会"


def test_a_gb2312_body_with_gbk_only_characters_is_read_as_gb18030():
    """声明 gb2312 的邮件里常有 GBK 才有的字 (镕), gb2312 解码器读不了; 按它的超集 gb18030 读."""
    raw = _mail("Subject: x\nContent-Type: text/plain; charset=gb2312\n", "朱镕基".encode("gbk"))
    assert email_text(raw) == "Subject: x\n\n朱镕基"


def test_an_unencoded_8bit_subject_is_read_with_the_fallback_encoding():
    raw = b"Subject: " + "会议通知".encode("gb18030") + b"\r\n\r\nbody\r\n"
    assert email_text(raw, fallback="gb18030") == "Subject: 会议通知\n\nbody"


def test_a_body_labelled_base64_that_is_not_base64_is_read_as_it_is():
    """trec06c 的正文常常已经解过码, 信头却还写着 base64; 照信头解就只剩空串."""
    raw = _mail("Subject: x\nContent-Type: text/plain; charset=gb2312\nContent-Transfer-Encoding: base64\n",
                "讲的是孔子后人的故事".encode("gbk"))
    assert email_text(raw) == "Subject: x\n\n讲的是孔子后人的故事"


def test_real_base64_is_still_decoded():
    raw = _mail("Subject: x\nContent-Type: text/plain; charset=utf-8\nContent-Transfer-Encoding: base64\n",
                "aGVsbG8gd29ybGQ=\n")
    assert email_text(raw) == "Subject: x\n\nhello world"


def test_iso_2022_jp_escape_sequences_without_a_declared_charset_are_decoded():
    jp = "小次郎".encode("iso-2022-jp")
    assert jp.isascii(), "ISO-2022-JP 是 7 位的, 不声明字符集时看上去就是 ASCII 加 ESC"
    raw = b"Subject: " + jp + b"\r\n\r\n" + jp + b"\r\n"
    assert email_text(raw) == "Subject: 小次郎\n\n小次郎"


def test_control_characters_other_than_newline_and_tab_are_dropped():
    raw = _mail("Subject: a\x07b\n", "c\x00d\x1be\x7ff\n")
    assert email_text(raw) == "Subject: ab\n\ncdef"


def test_a_multipart_header_over_a_flattened_body_reads_the_body_as_it_is():
    """trec06c 常见: 信头还是 multipart, 正文已被拍平成一段文字, 找不到分段边界."""
    raw = _mail("Subject: x\nContent-Type: multipart/related; boundary=\"----=_NextPart_000\"\n",
                "\n   非财务经理的财务管理\n\n   [课程背景]\n".encode("gb18030"))
    assert email_text(raw, fallback="gb18030") == "Subject: x\n\n非财务经理的财务管理\n\n[课程背景]"


def test_a_flattened_multipart_body_that_is_html_loses_its_tags():
    raw = _mail("Subject: x\nContent-Type: multipart/alternative; boundary=\"B\"\n",
                "<html><body><p>Hello&nbsp;there</p></body></html>\n")
    assert email_text(raw) == "Subject: x\n\nHello there"


def test_blank_line_runs_collapse_to_one_and_trailing_spaces_go():
    raw = _mail("Subject: S\n", "line one   \n\n\n\n  \nline two\t\n")
    assert email_text(raw) == "Subject: S\n\nline one\n\nline two"


def test_a_long_body_is_cut_to_max_body_characters():
    raw = _mail("Subject: Long\n", "word " * 1000)
    t = email_text(raw, max_body=100)
    body = t.split("\n\n", 1)[1]
    assert len(body) <= 100 and body.startswith("word word")


def test_an_email_with_neither_subject_nor_body_is_empty():
    assert email_text(_mail("From: a@b.c\n", "\n\n")) == ""


# --------------------------------------------------------------------------
# 去重
# --------------------------------------------------------------------------


def test_dedupe_keeps_the_first_copy_and_drops_texts_seen_under_both_labels():
    texts = ["a", "b", "a", "c", "b", "d"]
    labels = [0, 1, 0, 1, 0, 0]
    assert dedupe(texts, labels) == (["a", "c", "d"], [0, 1, 0])


def test_garbled_counts_replacement_and_private_use_characters():
    assert garbled("clean text") == 0.0
    assert garbled("ab\ufffd\ue42d") == 0.5


# --------------------------------------------------------------------------
# 语料 (要原始包在盘上)
# --------------------------------------------------------------------------


def _check_corpus(c, name):
    assert c.name == name
    assert len(c.texts) == len(c.labels) == len(c.ids)
    assert set(c.labels) == {0, 1}
    assert len(set(c.texts)) == len(c.texts), "去过重, 一段文本只剩一份"
    assert all(t and garbled(t) <= MAX_GARBLED for t in c.texts), "空文本与乱码文本都已丢掉"


def test_sms_is_the_uci_file_with_html_entities_restored_and_duplicates_dropped():
    c = load_spam("sms")
    _check_corpus(c, "sms")
    assert not any("&lt;" in t or "&gt;" in t or "&amp;" in t for t in c.texts)
    assert "Ok lar... Joking wif u oni..." in c.texts
    n_spam = sum(c.labels)
    assert (len(c.texts) - n_spam, n_spam) == (4518, 642), \
        "原文件 4827 ham / 747 spam; 实体还原、空白并一后去重剩 4518 / 642"


def test_enron_reads_the_raw_mailboxes_as_subject_plus_body_without_headers():
    c = load_spam("enron")
    _check_corpus(c, "enron")
    ham_boxes = {i.split("/")[0] for i, lab in zip(c.ids, c.labels) if lab == 0}
    spam_boxes = {i.split("/")[0] for i, lab in zip(c.ids, c.labels) if lab == 1}
    assert ham_boxes == {"beck-s", "farmer-d", "kaminski-v", "kitchen-l", "lokay-m", "williams-w3"}
    assert spam_boxes == {"BG", "GP", "SH"}
    heady = sum(t.startswith(("Message-ID:", "Return-Path:", "Received:")) for t in c.texts)
    assert heady < 0.001 * len(c.texts), f"只有正文里本身贴着一段信头的畸形邮件才会这样开头, 得到 {heady}"
    n_spam = sum(c.labels)
    assert 15000 < len(c.texts) - n_spam <= 19088 and 25000 < n_spam <= 32988, (len(c.texts) - n_spam, n_spam)


def test_trec_labels_come_from_the_full_index():
    """同一轮群发的 spam 只有信头不同, 去掉信头就是同一段文本: trec06p 的 24912 封 spam 只有约 6250 段不同的,
    trec06c 的 42854 封 spam 只有约 10450 段 (另有约 5850 封是原件里双字节字被截断后整段错位的乱码, 丢掉).
    下限按实测给, 防的是解析把大批正文读丢."""
    for name, n_ham, n_spam, min_ham, min_spam in (("trec06p", 12910, 24912, 12000, 6000),
                                                    ("trec07p", 25220, 50199, 24000, 30000),
                                                    ("trec06c", 21766, 42854, 20000, 10000)):
        c = load_spam(name)
        _check_corpus(c, name)
        got_ham, got_spam = len(c.texts) - sum(c.labels), sum(c.labels)
        assert min_ham < got_ham <= n_ham and min_spam < got_spam <= n_spam, (name, got_ham, got_spam)


def test_trec06c_bodies_are_mostly_there():
    c = load_spam("trec06c")
    subject_only = sum(1 for t in c.texts if "\n\n" not in t)
    assert subject_only < 0.02 * len(c.texts), subject_only


def test_trec07p_keeps_a_known_ham_mail_under_its_own_id():
    c = load_spam("trec07p")
    i = c.ids.index("inmail.934")
    assert c.labels[i] == 0 and c.texts[i].startswith("Subject: [sugar] Error starting up sugar\n\n")


def test_telegram_is_the_hand_tagged_posts_of_telegram_research_v1():
    """v1 的 adv_tags 表: 5270 条帖子人工标了是不是广告 (3167 是 / 2103 否). 帖子正文来自 posts 表, id 是帖子 id."""
    c = load_spam("telegram")
    _check_corpus(c, "telegram")
    n_spam = sum(c.labels)
    assert 1900 < len(c.texts) - n_spam <= 2103 and 2300 < n_spam <= 3167, (len(c.texts) - n_spam, n_spam)
    i = c.ids.index("139236")
    assert c.labels[i] == 1 and c.texts[i].startswith("?کــــانــــال شارژ رایگان\nکانال اطلاع از طرحها")
    i = c.ids.index("9758")
    assert c.labels[i] == 0 and c.texts[i].startswith("نصیحت چندم:\nسعی کنید")


def test_telegram_texts_are_cut_to_max_body():
    """Telegram 一条消息最长 4096 字符, 与邮件一样截到 MAX_BODY."""
    assert max(len(t) for t in load_spam("telegram").texts) <= MAX_BODY


def test_trec06c_is_read_as_chinese():
    c = load_spam("trec06c")
    han = sum(1 for t in c.texts if any("\u4e00" <= ch <= "\u9fff" for ch in t))
    assert han > 0.9 * len(c.texts), han


def test_the_parsed_texts_are_cached_and_reread_identically():
    a = load_spam("sms")
    assert cache_path("sms").exists()
    assert load_spam("sms") == a


def test_a_source_file_with_the_wrong_md5_is_refused():
    with tempfile.TemporaryDirectory() as d:
        (pathlib.Path(d) / "sms-spam").mkdir()
        (pathlib.Path(d) / "sms-spam" / "sms+spam+collection.zip").write_bytes(b"not the zip")
        try:
            load_spam("sms", data_dir=d)
        except RuntimeError as e:
            assert "md5" in str(e)
        else:
            raise AssertionError("wrong md5 was accepted")


def test_every_source_file_has_a_download_url():
    """每个原始包都有下载地址. Enron-Spam 的原站证书链不全 (urllib 验不过), TREC 的原站已 404, 两者都取 Wayback 存档,
    带 id_ 取原始字节; 时间戳钉死, 不经重定向."""
    rels = {rel for files in SOURCES.values() for rel in files}
    assert set(URLS) == rels
    assert all(u.startswith("https://") for u in URLS.values())
    assert all("web.archive.org/web/" in u and "id_/" in u for r, u in URLS.items() if r.startswith(("enron", "trec")))


def test_a_missing_source_file_is_downloaded_from_its_url():
    """空目录里第一次读 sms: 从它的地址下原始包 (换成拷贝盘上那份), 读出与默认目录相同的语料."""
    rel = "sms-spam/sms+spam+collection.zip"
    urls = []

    def fake(url, dest):
        urls.append(url)
        shutil.copy(DEFAULT_DATA_DIR / rel, dest)

    with tempfile.TemporaryDirectory() as d:
        real, urllib.request.urlretrieve = urllib.request.urlretrieve, fake
        try:
            c = load_spam("sms", data_dir=d)
        finally:
            urllib.request.urlretrieve = real
        assert (pathlib.Path(d) / rel).exists()
    assert urls == [URLS[rel]]
    assert c == load_spam("sms")


if __name__ == "__main__":
    run(globals())
