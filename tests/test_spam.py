"""decidophobia.data.spam 的测试: 一封原始邮件读成给模型看的一段文本.

跑:  PYTHONPATH=src .venv/bin/python tests/test_spam.py
"""

from _runner import run
from decidophobia.data.spam import email_text


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


if __name__ == "__main__":
    run(globals())
