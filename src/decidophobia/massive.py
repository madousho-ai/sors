"""MASSIVE (Amazon, en-US) 适配. test 分区是留出评估 (scripts/eval-massive.py), train 分区可进训练 (--dataset massive).

60 个 intent、18 个 scenario, 语音助手指令 (闹钟 / 灯 / 音乐 / 天气 ...), 与 Banking77 零重叠.
不训它时回答: 在 Banking77 + BoolQ 上训的 LoRA, 换一个领域的意图分类还会不会读菜单.

数据: https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz (39.5MB, 51 locale)
取 1.0/data/en-US.jsonl, 分区 test 2974 条 / train 11514 条. tarball 不自动下载, md5 钉死.
类 id 按原始 intent 名字母序编, 与 banking77 同一约定.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import tarfile

from decidophobia.data import LabeledSet

TARBALL = "amazon-massive-dataset-1.0.tar.gz"
TARBALL_MD5 = "92fe0007628b31ca02c7bf4035a883e7"
MEMBER = "1.0/data/en-US.jsonl"
CONTEXT_LABEL = "Voice command"

# 原始名里粘连在一起的复合词. 品牌名 hue (飞利浦灯) / wemo (智能插座) 保留.
_COMPOUND = {
    "lightchange": "light change", "lightdim": "light dim", "lightoff": "light off",
    "lighton": "light on", "lightup": "light up", "createoradd": "create or add",
    "sendemail": "send email", "addcontact": "add contact", "querycontact": "query contact",
}


def humanize_intent(raw: str) -> str:
    """'iot_hue_lightchange' -> 'iot: hue light change'. 第一段是 scenario, 其余是动作."""
    scenario, _, action = raw.partition("_")
    words = [_COMPOUND.get(w, w) for w in action.split("_")]
    return f"{scenario}: {' '.join(words)}"


def _read_rows(cache_dir) -> list[dict]:
    cache_dir = pathlib.Path(cache_dir)
    jsonl = cache_dir / "en-US.jsonl"
    if not jsonl.exists():
        tb = cache_dir / TARBALL
        got = hashlib.md5(tb.read_bytes()).hexdigest()
        if got != TARBALL_MD5:
            raise RuntimeError(f"{tb}: md5 {got} != {TARBALL_MD5}")
        with tarfile.open(tb) as t:
            jsonl.write_bytes(t.extractfile(MEMBER).read())
    with jsonl.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_massive(cache_dir="data/massive", partition: str = "test") -> LabeledSet:
    rows = _read_rows(cache_dir)
    raw_names = sorted({r["intent"] for r in rows})
    idx = {c: i for i, c in enumerate(raw_names)}
    part = [r for r in rows if r["partition"] == partition]
    return LabeledSet(
        queries=[r["utt"] for r in part],
        labels=[idx[r["intent"]] for r in part],
        names={i: humanize_intent(c) for c, i in idx.items()},
        context_label=CONTEXT_LABEL,
    )
