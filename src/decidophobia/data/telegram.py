"""arashdn/telegram-research 的 v2: Telegram 公开频道里走红的消息, 波斯语为主, 每条人工标了类别与情感.
(v1 那张广告 / 非广告标注表是二元的 spam 语料, 在 decidophobia.data.spam 里, 名字 "telegram".)

原始包 data/telegram/tg_v2_public.zip (作者 README 里的 Dropbox 链接, 54MB, md5 钉死) 里是一份 MySQL dump,
第一次用时自动下载 (见 decidophobia.data.download).
只读其中三张表: viral_messages (18566 条带标注的消息), tags (24 个类别), super_tags (类别上面的 8 个大类).
posts 表的 45 万条消息没有标注, 不读.

一条消息的文本取 message, 为空就取 caption (配图或视频的说明文字); 两个字段不会同时有字.
爬虫把换行存成了两个字符 \\n, 这里还原成换行. 表情符号在原库里已经丢了, 多数变成了 "?", 无从复原, 原样留着.
同一段文本常被好几个频道转发, 每份拷贝各标各的, 而且常常标得不一样 (同文多份里类别全一致的只有一半).
于是按文本并成一条, 类别与情感记成计数: {类别 id: 几份拷贝标了它}. 用硬标签还是按计数给软标签, 由出题的一方决定.

sentiment 只有 -1 / 0 / 1 (负面 / 中性 / 正面); 库里各有一条 2 与 -2, 那份拷贝的情感不计.
解析结果缓存在原始包旁边的 viral.p<PARSE_VERSION>.json.
"""

from __future__ import annotations

import io
import json
import pathlib
import zipfile
from dataclasses import dataclass

from decidophobia.data.download import fetch
from decidophobia.data.mysqldump import dump_tables
from decidophobia.data.spam import DEFAULT_DATA_DIR, MAX_BODY, tidy

ZIP = "telegram/tg_v2_public.zip"
ZIP_MD5 = "302c13511699f51aed674b891f3e795e"
URL = "https://www.dropbox.com/s/sokcxz35e4ta91l/tg_v2_public.zip?dl=1"
MEMBER = "tg_v2_public.sql"
PARSE_VERSION = 1
SENTIMENTS = (-1, 0, 1)


# --------------------------------------------------------------------------
# 帖子
# --------------------------------------------------------------------------


def post_text(message: str | None, caption: str | None, max_body: int = MAX_BODY) -> str:
    """一条帖子给模型看的文本: message, 为空就用 caption. 字面的 \\n 还原成换行, 再按 spam.tidy 收拾, 截到 max_body 个字符."""
    s = (message or caption or "").replace("\\n", "\n")
    return tidy(tidy(s)[:max_body])


def merge_copies(posts: list[dict]) -> list[dict]:
    """posts 每项 {"id", "text", "tag", "sentiment"}. 同一段文本并成一条 {"text", "ids", "tags", "sentiments"},
    tags / sentiments 是 {值: 几份拷贝标了它}; 顺序按每段文本第一次出现. 空文本丢掉, 超出 -1..1 的情感不计."""
    merged: dict[str, dict] = {}
    for p in posts:
        if not p["text"]:
            continue
        m = merged.setdefault(p["text"], {"text": p["text"], "ids": [], "tags": {}, "sentiments": {}})
        m["ids"].append(p["id"])
        m["tags"][p["tag"]] = m["tags"].get(p["tag"], 0) + 1
        if p["sentiment"] in SENTIMENTS:
            m["sentiments"][p["sentiment"]] = m["sentiments"].get(p["sentiment"], 0) + 1
    return list(merged.values())


# --------------------------------------------------------------------------
# 语料
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TelegramCorpus:
    posts: list[dict]  # merge_copies 的输出
    tags: dict[int, tuple[str, int]]  # 类别 id -> (库里的名字, 大类 id)
    super_tags: dict[int, str]  # 大类 id -> 库里的名字


def cache_path(data_dir=DEFAULT_DATA_DIR) -> pathlib.Path:
    return pathlib.Path(data_dir) / "telegram" / f"viral.p{PARSE_VERSION}.json"


def _int_keys(d: dict) -> dict:
    return {int(k): v for k, v in d.items()}


def _from_json(obj: dict) -> TelegramCorpus:
    posts = [{**p, "tags": _int_keys(p["tags"]), "sentiments": _int_keys(p["sentiments"])} for p in obj["posts"]]
    return TelegramCorpus(posts, {int(k): tuple(v) for k, v in obj["tags"].items()}, _int_keys(obj["super_tags"]))


def _parse(zip_path: pathlib.Path) -> TelegramCorpus:
    with zipfile.ZipFile(zip_path) as z, io.TextIOWrapper(z.open(MEMBER), encoding="utf-8") as f:
        tables = dump_tables(f, ("viral_messages", "tags", "super_tags"))
    # viral_messages 的列: id, fwd_from, from_id, message, caption, date, views, sender_title, tag, sentiment, is_viral
    posts = [{"id": r[0], "text": post_text(r[3], r[4]), "tag": r[8], "sentiment": r[9]}
             for r in tables["viral_messages"]]
    return TelegramCorpus(merge_copies(posts), {r[0]: (r[1], r[2]) for r in tables["tags"]},
                          {r[0]: r[1] for r in tables["super_tags"]})


def load_telegram_viral(data_dir=DEFAULT_DATA_DIR) -> TelegramCorpus:
    """第一次读原始包 (没有就下载, 核对 md5) 并写缓存, 之后读缓存."""
    cache = cache_path(data_dir)
    if cache.exists():
        return _from_json(json.loads(cache.read_text(encoding="utf-8")))
    c = _parse(fetch(pathlib.Path(data_dir) / ZIP, URL, ZIP_MD5))
    cache.write_text(json.dumps({"tags": c.tags, "super_tags": c.super_tags, "posts": c.posts}, ensure_ascii=False),
                     encoding="utf-8")
    return _from_json(json.loads(cache.read_text(encoding="utf-8")))
