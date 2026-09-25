"""Banking77 适配: 读上游 CSV (钉 commit + md5), 给出 queries / labels / names.

菜单上显示的文字由 labels 选: raw 原始 label 名 (默认), desc 写好的 description, 见 decidophobia.label_names.
与 scripts/baseline-banking77.py 的 load_split 同源. 那个脚本是独立产物, 不从包里 import.
"""

from __future__ import annotations

import csv
import hashlib
import pathlib
import urllib.request

from decidophobia.data import LabeledSet
from decidophobia.label_names import label_names

DATA_COMMIT = "9d081458ff52e53cf7e848f414e6e9344e4e6696"
DATA_FILES = {
    "train": ("cec64185f4197906aabce0781ef9a19b", 10003),
    "test": ("8dcd9dc31b686c75ec1f24bf23c140cb", 3080),
}


def _fetch(split: str, cache_dir) -> list[tuple[str, str]]:
    want_md5, want_n = DATA_FILES[split]
    path = pathlib.Path(cache_dir) / f"{split}.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = (
            f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
            f"{DATA_COMMIT}/banking_data/{split}.csv"
        )
        urllib.request.urlretrieve(url, path)
    got = hashlib.md5(path.read_bytes()).hexdigest()
    if got != want_md5:
        raise RuntimeError(f"{path}: md5 {got} != {want_md5}")
    with path.open(encoding="utf-8") as f:
        rows = [(r["text"], r["category"]) for r in csv.DictReader(f)]
    if len(rows) != want_n:
        raise RuntimeError(f"{path}: {len(rows)} rows != {want_n}")
    return rows


def load_banking77(cache_dir="data/banking77", labels: str = "raw") -> tuple[LabeledSet, LabeledSet]:
    """返回 (train, test). 类 id 按原始 label 名字母序编, 两个 split 共用同一张表; labels 见 label_names."""
    tr, te = _fetch("train", cache_dir), _fetch("test", cache_dir)
    raw_names = sorted({c for _, c in tr} | {c for _, c in te})
    idx = {c: i for i, c in enumerate(raw_names)}
    names = label_names("banking77", raw_names, labels)

    def mk(rows):
        return LabeledSet(queries=[t for t, _ in rows], labels=[idx[c] for _, c in rows], names=names,
                          context_label="Customer message")

    return mk(tr), mk(te)
