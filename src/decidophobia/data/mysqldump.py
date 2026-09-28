"""从 mysqldump 导出的 .sql 文本里读表的行, 不起 MySQL. arashdn/telegram-research 的 v1 与 v2 都是这种导出.

只认 mysqldump 默认的写法: 每张表的数据是若干行 "INSERT INTO `表名` VALUES (...),(...);", 一行一条语句.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_STRING = re.compile(r"'((?:[^'\\]|\\.|'')*)'", re.S)
_BARE = re.compile(r"[^,)]+")
_ESCAPE = re.compile(r"\\(.)|''", re.S)
_ESCAPES = {"0": "\0", "n": "\n", "r": "\r", "t": "\t", "Z": "\x1a", "b": "\b"}
_INSERT = re.compile(r"INSERT INTO `(\w+)` VALUES ")


def _unescape(s: str) -> str:
    return _ESCAPE.sub(lambda m: "'" if m.group(1) is None else _ESCAPES.get(m.group(1), m.group(1)), s)


def _bare(tok: str):
    if tok == "NULL":
        return None
    return float(tok) if any(c in tok for c in ".eE") else int(tok)


def sql_rows(line: str) -> list[list]:
    """mysqldump 的一行 "INSERT INTO `t` VALUES (...),(...);" -> 每个括号一行, 字符串去掉转义, 数字转成 int / float."""
    i = line.index(" VALUES ") + len(" VALUES ")
    rows = []
    while True:
        if line[i] != "(":
            raise ValueError(f"expected '(' at {i}: {line[i:i + 40]!r}")
        i += 1
        row = []
        while True:
            if line[i] == "'":
                m = _STRING.match(line, i)
                row.append(_unescape(m.group(1)))
            else:
                m = _BARE.match(line, i)
                row.append(_bare(m.group()))
            i = m.end()
            if line[i] == ",":
                i += 1
                continue
            if line[i] != ")":
                raise ValueError(f"expected ',' or ')' at {i}: {line[i:i + 40]!r}")
            i += 1
            break
        rows.append(row)
        if line[i] == ";":
            return rows
        if line[i] != ",":
            raise ValueError(f"expected ',' or ';' at {i}: {line[i:i + 40]!r}")
        i += 1


def dump_tables(lines: Iterable[str], names: tuple[str, ...]) -> dict[str, list[list]]:
    """整份 dump 按行读, 只解析 names 里那几张表的 INSERT, 同一张表的各行首尾相接. 其余的表连解析都不做."""
    out: dict[str, list[list]] = {n: [] for n in names}
    for line in lines:
        m = _INSERT.match(line)
        if m and m.group(1) in out:
            out[m.group(1)] += sql_rows(line.rstrip("\n"))
    return out
