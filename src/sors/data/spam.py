"""spam 语料读成 (文本, 是否 spam): SMS Spam Collection / Enron-Spam / TREC Public Spam Corpus / Telegram 广告标注.

语料 (load_spam 的名字), 原始包放在 data/ 下; 第一次用到时从 URLS 下载 (md5 钉死, 见 sors.data.download):
  sms       data/sms-spam/sms+spam+collection.zip        UCI, 5574 条英文短信
  enron     data/enron-spam/raw/{ham,spam}/*.tar.gz      Enron-Spam 原始形态: 6 个 Enron 员工邮箱的 ham + 3 个来源的 spam
  trec06p   data/trec-spam/trec06p.tgz                   TREC 2006 英文, 37822 封
  trec06c   data/trec-spam/trec06c.tgz                   TREC 2006 中文, 64620 封
  trec07p   data/trec-spam/trec07p.tgz                   TREC 2007, 75419 封
  telegram  data/telegram/tg_public.zip                  arashdn/telegram-research v1 (Dropbox), 波斯语频道帖子,
                                                         adv_tags 表里 5270 条人工标了是不是广告
TREC 的三个包取自 web.archive.org 存的 plg.uwaterloo.ca 原件 (官网已 404); TREC 2005 那份存档里没有文件本体.
Enron-Spam 的原站 www2.aueb.gr 还在, 但证书链不全, urllib 验不过, 同样取 web.archive.org 的存档 (与原件 md5 相同).

邮件统一成同一个样子 (email_text): 一行 "Subject: <标题>", 空一行, 然后是正文. 其余信头全部丢掉 ——
Received / Message-ID / 发件服务器这些在三个语料里各有各的来源特征, 留着就等于把答案写在题面上.
正文取 text/plain, 没有才取 text/html 并去掉标签; 正文截到 max_body 个字符.
短信与 Telegram 帖子没有标题, 文本就是正文本身, 同样收拾空白、截到 max_body.
之后丢掉空文本和乱码文本 (garbled 超过 MAX_GARBLED), 再去重 (dedupe).
解析一遍要几十秒, 结果缓存在原始包旁边的 <名字>.v<PARSE_VERSION>.jsonl; 改了解析规则就把 PARSE_VERSION 加一.

类 id 0 = ham, 1 = spam.
"""

from __future__ import annotations

import email
import email.header
import html
import io
import json
import pathlib
import re
import tarfile
import zipfile
from dataclasses import dataclass
from email import policy
from html.parser import HTMLParser

from sors.data.download import fetch
from sors.data.mysqldump import dump_tables

MAX_BODY = 2000  # 正文最多几个字符
MAX_GARBLED = 0.01  # 乱码字符 (U+FFFD 与私用区) 占比超过它的文本丢掉
PARSE_VERSION = 1
DEFAULT_DATA_DIR = pathlib.Path(__file__).resolve().parents[3] / "data"

# 相对 data/ 的路径 -> md5
SOURCES = {
    "sms": {"sms-spam/sms+spam+collection.zip": "ab53f9571d479ee677e7b283a06a661a"},
    "enron": {
        "enron-spam/raw/ham/beck-s.tar.gz": "13bc73f1ce9e0e33f604fdf611d0321a",
        "enron-spam/raw/ham/farmer-d.tar.gz": "a71d306c4ac5e52e6718b3cc03f2acdb",
        "enron-spam/raw/ham/kaminski-v.tar.gz": "e78c0b848349618a313dd9fbdadaf36f",
        "enron-spam/raw/ham/kitchen-l.tar.gz": "2eb263cfbcc3dc8cab89c2c57b9d960e",
        "enron-spam/raw/ham/lokay-m.tar.gz": "6248a755bc56a15093c7f2260339432d",
        "enron-spam/raw/ham/williams-w3.tar.gz": "25fd530e2e49940a5c3b27734c67c9ec",
        "enron-spam/raw/spam/BG.tar.gz": "91a94c1206301ee2f9c3f2b03328e52f",
        "enron-spam/raw/spam/GP.tar.gz": "f8a94e42e3f4ee847336b7ae22f0dc5c",
        "enron-spam/raw/spam/SH.tar.gz": "5ab5360072b34bc290517ce0c13d0975",
    },
    "trec06p": {"trec-spam/trec06p.tgz": "882d5de429562adf9071c130ddbf0936"},
    "trec06c": {"trec-spam/trec06c.tgz": "655d7e7a58f2b8f0d7382ebb16ae23df"},
    "trec07p": {"trec-spam/trec07p.tgz": "59c3df3efeb2fbd23babc18136bd466a"},
    "telegram": {"telegram/tg_public.zip": "f07268981fe1d544cceeb2c757ef38ed"},
}
CORPORA = tuple(SOURCES)

# 相对 data/ 的路径 -> 下载地址. Wayback 的地址带 id_, 取存档的原始字节; 时间戳是每个文件自己那份存档的
_ENRON = "https://web.archive.org/web/{}id_/https://www2.aueb.gr/users/ion/data/enron-spam/{}.tar.gz"
_TREC = "https://web.archive.org/web/{}id_/https://plg.uwaterloo.ca/cgi-bin/cgiwrap/gvcormac/{}.tgz"
URLS = {
    "sms-spam/sms+spam+collection.zip": "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
    **{f"enron-spam/{p}.tar.gz": _ENRON.format(ts, p) for p, ts in (
        ("raw/ham/beck-s", "20260213112247"), ("raw/ham/farmer-d", "20260213112251"),
        ("raw/ham/kaminski-v", "20260213112243"), ("raw/ham/kitchen-l", "20260213112249"),
        ("raw/ham/lokay-m", "20260213112243"), ("raw/ham/williams-w3", "20260213112245"),
        ("raw/spam/BG", "20260213112241"), ("raw/spam/GP", "20260213112243"), ("raw/spam/SH", "20260213112253"))},
    "trec-spam/trec06p.tgz": _TREC.format("20250819075600", "trec06p"),
    "trec-spam/trec06c.tgz": _TREC.format("20250819075650", "trec06c"),
    "trec-spam/trec07p.tgz": _TREC.format("20250813102119", "trec07p"),
    "telegram/tg_public.zip": "https://www.dropbox.com/s/szcjfo5k4cxycxz/tg_public.zip?dl=1",
}
FALLBACK = {"trec06c": "gb18030"}  # 没声明字符集的正文先按它读

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


def dedupe(texts: list[str], labels: list[int]) -> tuple[list[str], list[int]]:
    """同一段文本只留第一份; 同一段文本在 ham 与 spam 两边都出现过的, 一份不留."""
    kept = _dedupe(texts, labels, texts)
    return [t for t, _, _ in kept], [lab for _, lab, _ in kept]


def _dedupe(texts, labels, ids) -> list[tuple[str, int, str]]:
    seen: dict[str, set[int]] = {}
    for t, lab in zip(texts, labels, strict=True):
        seen.setdefault(t, set()).add(lab)
    out, kept = [], set()
    for t, lab, i in zip(texts, labels, ids, strict=True):
        if len(seen[t]) > 1 or t in kept:
            continue
        kept.add(t)
        out.append((t, lab, i))
    return out


_GARBLE = re.compile("[\ufffd\ue000-\uf8ff]")


def garbled(s: str) -> float:
    """乱码字符 (解码失败留下的 U+FFFD, 以及错位的双字节读出来的私用区字) 占全文的比例."""
    return len(_GARBLE.findall(s)) / len(s) if s else 0.0


# --------------------------------------------------------------------------
# 语料
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SpamCorpus:
    name: str
    texts: list[str]
    labels: list[int]  # 0 = ham, 1 = spam
    ids: list[str]  # 每条在原始包里的出处: sms 是行号, enron 是 <邮箱>/<路径>, trec 是 data/ 下的文件名, telegram 是帖子 id


def _checked(data_dir: pathlib.Path, rel: str, md5: str) -> pathlib.Path:
    """data/ 下的原始包, 没有就从 URLS 下载; 两种情形都核对 md5."""
    return fetch(data_dir / rel, URLS[rel], md5)


def _sms(data_dir: pathlib.Path):
    (rel, md5), = SOURCES["sms"].items()
    with zipfile.ZipFile(_checked(data_dir, rel, md5)) as z:
        lines = z.read("SMSSpamCollection").decode("utf-8").splitlines()
    for n, line in enumerate(lines, 1):
        lab, text = line.split("\t", 1)
        yield tidy(html.unescape(text)), int(lab == "spam"), str(n)


def _tar_files(path: pathlib.Path):
    with tarfile.open(path) as t:
        for m in t:
            if m.isfile():
                yield m.name, t.extractfile(m).read()


def _enron(data_dir: pathlib.Path):
    for rel, md5 in SOURCES["enron"].items():
        lab = int("/spam/" in rel)
        for name, raw in _tar_files(_checked(data_dir, rel, md5)):
            yield email_text(raw), lab, name


def _trec(data_dir: pathlib.Path, corpus: str):
    (rel, md5), = SOURCES[corpus].items()
    label_of: dict[str, int] = {}
    texts: dict[str, str] = {}
    for name, raw in _tar_files(_checked(data_dir, rel, md5)):
        if name == f"{corpus}/full/index":
            for line in raw.decode().splitlines():
                lab, p = line.split()  # "spam ../data/inmail.1"
                label_of[p.split("data/", 1)[1]] = int(lab == "spam")
        elif "/data/" in name:
            texts[name.split("/data/", 1)[1]] = email_text(raw, fallback=FALLBACK.get(corpus))
    for i in sorted(label_of, key=_natural):  # 按投递顺序
        yield texts[i], label_of[i], i


def _natural(s: str):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", s)]


def _telegram(data_dir: pathlib.Path):
    """v1 的 adv_tags (post_id, is_adv) 配 posts 表的 body. posts 的列: id, tg_id, flags, date, body, from, to, org_messager.
    body 里的换行在 dump 里是正常的 \\n 转义, 读出来就是换行 (v2 那种字面 \\n 在 v1 里没有).
    users 表带手机号, 不解析."""
    (rel, md5), = SOURCES["telegram"].items()
    with zipfile.ZipFile(_checked(data_dir, rel, md5)) as z, io.TextIOWrapper(z.open("tg_public.sql"),
                                                                               encoding="utf-8") as f:
        tables = dump_tables(f, ("adv_tags", "posts"))
    body = {r[0]: r[4] for r in tables["posts"]}
    for post_id, is_adv in tables["adv_tags"]:  # 按标注表的顺序
        yield tidy(tidy(body[post_id] or "")[:MAX_BODY]), int(is_adv), str(post_id)


def cache_path(name: str, data_dir=DEFAULT_DATA_DIR) -> pathlib.Path:
    rel = next(iter(SOURCES[name]))
    return pathlib.Path(data_dir) / rel.split("/")[0] / f"{name}.v{PARSE_VERSION}.jsonl"


_READERS = {"sms": _sms, "enron": _enron, "telegram": _telegram}


def load_spam(name: str, data_dir=DEFAULT_DATA_DIR) -> SpamCorpus:
    """一个语料 (CORPORA 之一): 解析、丢空与乱码、去重之后的全部文本. 第一次读原始包并写缓存, 之后读缓存."""
    if name not in SOURCES:
        raise ValueError(f"unknown spam corpus {name!r}; expected one of {CORPORA}")
    data_dir = pathlib.Path(data_dir)
    cache = cache_path(name, data_dir)
    if cache.exists():
        # 只按 \n 切: splitlines 还会在 U+2028、\x1c、\x85 这些字符处切, 把一条 JSON 切成两半
        rows = [json.loads(line) for line in cache.read_text(encoding="utf-8").split("\n") if line]
        return SpamCorpus(name, [r["text"] for r in rows], [r["label"] for r in rows], [r["id"] for r in rows])
    rows = _READERS[name](data_dir) if name in _READERS else _trec(data_dir, name)
    rows = [(t, lab, i) for t, lab, i in rows if t and garbled(t) <= MAX_GARBLED]
    kept = _dedupe([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows])
    buf = io.StringIO()
    for t, lab, i in kept:
        buf.write(json.dumps({"id": i, "label": lab, "text": t}, ensure_ascii=False) + "\n")
    cache.write_text(buf.getvalue(), encoding="utf-8")
    return SpamCorpus(name, [r[0] for r in kept], [r[1] for r in kept], [r[2] for r in kept])
