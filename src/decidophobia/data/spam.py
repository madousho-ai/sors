"""spam 语料的解析: 一封原始邮件 (RFC 822 字节) 读成给模型看的一段文本 (email_text).

邮件统一成同一个样子: 一行 "Subject: <标题>", 空一行, 然后是正文. 其余信头全部丢掉 ——
Received / Message-ID / 发件服务器这些在各个语料里各有各的来源特征, 留着就等于把答案写在题面上.
正文取 text/plain, 没有才取 text/html 并去掉标签; 正文截到 max_body 个字符.
"""

from __future__ import annotations

import email
import email.header
import re
from email import policy
from html.parser import HTMLParser

MAX_BODY = 2000  # 正文最多几个字符
_SKIP_TAGS = {"script", "style", "title"}
_BLOCK_TAGS = {"p", "br", "div", "tr", "li", "ul", "ol", "table", "h1", "h2", "h3", "h4", "h5", "h6", "hr",
               "blockquote", "pre", "center", "form"}


class _HTMLText(HTMLParser):
    """HTML 里浏览器会显示出来的字: 去掉标签、注释、script / style, 实体还原, 块级标签处换行."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self.skip += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")
        elif tag == "td":
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS:
            self.skip = max(0, self.skip - 1)
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def html_text(s: str) -> str:
    p = _HTMLText()
    p.feed(s)
    p.close()
    return "".join(p.parts)


_CONTROL = re.compile("[\x00-\x08\x0e-\x1f\x7f-\x9f]")


def tidy(s: str) -> str:
    """去掉控制字符 (换行与 tab 以外), 行内的连续空白并成一个空格、每行去掉首尾空白, 连续空行并成一行, 整段去掉首尾空行."""
    s = s.replace("\r\n", "\n").replace("\r", "\n").replace("\u2028", "\n").replace("\u2029", "\n")
    s = _CONTROL.sub("", s.replace("\xa0", " "))
    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in s.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


# 声明成 gb2312 / gbk 的中文邮件常混着超集里才有的字, 一律按最大的超集 gb18030 读.
# 声明成 latin-1 / ascii 的多半其实是 windows-1252 (引号、破折号落在 0x80-0x9f), 与浏览器同样处理
_CHARSET_ALIAS = {"gb2312": "gb18030", "gbk": "gb18030", "x-gbk": "gb18030", "euc-cn": "gb18030",
                  "gb_2312-80": "gb18030", "chinese": "gb18030",
                  "iso-8859-1": "cp1252", "latin-1": "cp1252", "latin1": "cp1252", "us-ascii": "cp1252",
                  "ascii": "cp1252"}


def decode_bytes(b: bytes, declared: str | None, fallback: str | None) -> str:
    """依次试: 声明的字符集、语料给的 fallback、utf-8, 第一个能完整解码的胜出. 带 ESC $ 的 (ISO-2022-JP, 7 位,
    声不声明都像 ASCII) 最先试它. 全都不行 (或声明的名字 Python 不认识, 如 DEFAULT / X-UNKNOWN)
    就用 fallback 或 latin-1 硬读, 坏字节换成 U+FFFD."""
    tries = ["iso-2022-jp"] if b"\x1b$" in b else []
    tries += [_CHARSET_ALIAS.get(declared.lower(), declared)] if declared else []
    tries += [fallback] if fallback else []
    tries.append("utf-8")
    for enc in tries:
        try:
            return b.decode(enc)
        except (LookupError, UnicodeDecodeError):
            continue
    return b.decode(fallback or "latin-1", "replace")


def _subject(msg, fallback: str | None) -> str:
    """Subject 信头. 编码词 (=?charset?b?...?=) 逐段按它的字符集解; 没编码直接塞的字节 (8 位的, 或 ISO-2022-JP 的 ESC 序列)
    按 decode_bytes 读."""
    raw = next((v for k, v in msg.raw_items() if k.lower() == "subject"), None)
    if raw is None:
        return ""
    b = raw.encode("ascii", "surrogateescape")  # message_from_bytes 把非 ASCII 字节藏成 surrogate, 这里还原
    if not b.isascii() or b"\x1b" in b:
        s = decode_bytes(b, None, fallback)
    else:
        try:
            s = "".join(chunk if isinstance(chunk, str) else decode_bytes(chunk, cs, fallback)
                        for chunk, cs in email.header.decode_header(raw))
        except Exception:  # 编码词写坏了: 原样保留那串 ASCII
            s = raw
    return tidy(re.sub(r"\s+", " ", s))


_BASE64 = re.compile(rb"[A-Za-z0-9+/=\s]*")


def _payload(part) -> bytes:
    """一个部分的正文字节. 信头说 base64 而内容明显不是 (trec06c 常见: 正文已经解过码, 信头没改) 就原样用.
    原样的字节要从 _payload 拿: get_payload(decode=False) 碰到 8 位字节会先按声明的字符集解成文本."""
    raw = part._payload
    if isinstance(raw, str) and part.get("Content-Transfer-Encoding", "").strip().lower() == "base64":
        b = raw.encode("utf-8", "surrogateescape")  # 解析时非 ASCII 字节藏成了 surrogate, 这里还原成原字节
        if not _BASE64.fullmatch(b):
            return b
    return part.get_payload(decode=True) or b""


def _decode(part, fallback: str | None) -> str:
    """一个 text/* 部分的正文: 解传输编码 (_payload), 再按 decode_bytes 的顺序解字符."""
    return decode_bytes(_payload(part), part.get_content_charset(), fallback)


_LOOKS_HTML = re.compile(r"<\s*(html|body|p|br|div|table|font|a)\b", re.IGNORECASE)


def email_text(raw: bytes, max_body: int = MAX_BODY, fallback: str | None = None) -> str:
    """一封原始邮件 (RFC 822 字节) -> "Subject: <标题>\\n\\n<正文>". 没有标题就只有正文, 没有正文就只有标题行.
    fallback: 没声明字符集 (或声明的读不通) 时先试的编码, 中文语料给 gb18030."""
    msg = email.message_from_bytes(raw, policy=policy.default)
    subject = _subject(msg, fallback)
    part = msg.get_body(preferencelist=("plain", "html"))
    is_html = part is not None and part.get_content_type() == "text/html"
    if part is None and (msg.get_content_maintype() == "text" or isinstance(msg._payload, str)):
        # text/enriched 之类; 或信头说 multipart 而正文已被拍平成一段文字 (trec06c). 当纯文本读, 像 HTML 就去标签
        part = msg
        is_html = bool(_LOOKS_HTML.search(msg._payload))
    body = ""
    if part is not None:
        body = _decode(part, fallback)
        if is_html:
            body = html_text(body)
        body = tidy(body)[:max_body].rstrip()
    head = f"Subject: {subject}" if subject else ""
    return "\n\n".join(x for x in (head, body) if x)
