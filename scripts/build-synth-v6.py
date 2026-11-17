#!/usr/bin/env python
"""写 synth-intents-v6 的训练名单 selection.json (逐道绑定, 外加 v5.25 每个文件的 sha256).

  PYTHONPATH=src .venv/bin/python scripts/build-synth-v6.py --datasets-dir ../sors-dataset           写
  PYTHONPATH=src .venv/bin/python scripts/build-synth-v6.py --datasets-dir ../sors-dataset --check   重算, 与文件不同就退出码 1

规则在 synth-intents-v6/spec.json: source (v5.25 目录名), classes {类: [数据家族]}, n {类: 绑定数}, eval_n, salt.
不进训练的材料在 synth-intents-v6/drop.json: contexts 直接列出; also 里的每个文件 (相对数据根) 的 contexts 也算.
选法见 sors.data.synth_v6.build_selection.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from sors.data.paths import add_datasets_argument, datasets_root
from sors.data.synth_v5 import load_synth_v5
from sors.data.synth_v6 import NAME, build_selection, source_sha256


def selection(root: pathlib.Path) -> dict:
    v6 = root / NAME
    spec = json.loads((v6 / "spec.json").read_text(encoding="utf-8"))
    drop_spec = json.loads((v6 / "drop.json").read_text(encoding="utf-8"))
    drop = set(drop_spec.get("contexts", []))
    for rel in drop_spec.get("also", []):
        drop |= set(json.loads((root / rel).read_text(encoding="utf-8"))["contexts"])
    src = root / spec["source"]
    keys = build_selection(load_synth_v5(src), spec, drop)
    return {"source": spec["source"], "source_sha256": source_sha256(src), "spec": spec, "dropped_contexts": len(drop),
            "bindings": [list(k) for k in keys]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_datasets_argument(ap)
    ap.add_argument("--check", action="store_true", help="重算并与 selection.json 比对, 不写文件")
    args = ap.parse_args(argv)
    root = datasets_root(args.datasets_dir)
    sel = selection(root)
    text = json.dumps(sel, ensure_ascii=False, separators=(",", ":")) + "\n"
    path = root / NAME / "selection.json"
    if args.check:
        same = path.exists() and path.read_text(encoding="utf-8") == text
        print(f"{path}: {'up to date' if same else 'differs from a fresh build'} ({len(sel['bindings'])} bindings)")
        return 0 if same else 1
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path}: {len(sel['bindings'])} bindings, {sel['dropped_contexts']} dropped materials")
    return 0


if __name__ == "__main__":
    sys.exit(main())
