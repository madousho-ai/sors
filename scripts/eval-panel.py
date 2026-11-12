#!/usr/bin/env python
"""评估面板: 把各来源的评估集导出成统一的「请求 + 标准答案」, 经 /v1/systemone 打分, 按评估集出表.

  # 1. 导出 (每个评估集一个 <名字>.jsonl, 外加 manifest.json 记条数与 sha256)
  PYTHONPATH=src .venv/bin/python scripts/eval-panel.py export --out data/panel \
      --task sors=../sors-dataset/<留出评估集>.jsonl --task json=../sors-dataset/json-eval/json-eval-v1.jsonl \
      --task zh=../sors-dataset/zh-eval/zh-eval-v1.jsonl \
      --task jevbench_easy=data/jevbench/easy.jsonl --task jevbench_original=data/jevbench/original.jsonl \
      --task jevbench_hard=data/jevbench/hard.jsonl \
      --simple ../sors-dataset/synth-simple-eval/synth-simple-eval.jsonl --massive 1000 --boolq 1000 \
      [--suite dindex=<Decision Index 题集的行文件 .jsonl.gz>]

  # 2. 打分 (服务先起好; 续跑跳过已答的, 报错的重发)
  PYTHONPATH=src .venv/bin/python scripts/eval-panel.py run --panel data/panel \
      --base-url http://127.0.0.1:8000 --model <服务名> --out runs/panel/<模型>.jsonl

  # 3. 出表 (几个结果文件并排比)
  PYTHONPATH=src .venv/bin/python scripts/eval-panel.py report runs/panel/a.jsonl runs/panel/b.jsonl

--task 读 JevBench 题目格式 (state, question, labels, expected) 的 jsonl, 名字自己起; 路径都从命令行给,
本仓库的代码里不写评估集的目录名. --massive / --boolq 给条数: 按题号的 sha256 排序后取前 N 条, 每次导出相同.
计分规则见 sors.evaluation.panel.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

from sors.evaluation.panel import (item_from_simple, item_from_suite_row, item_from_task, items_from_labeled,
                                   load_results, run_items, summarize)


def _read_jsonl(path) -> list[dict]:
    path = pathlib.Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _named(spec: str, flag: str) -> tuple[str, str]:
    if "=" not in spec:
        raise SystemExit(f"{flag} wants NAME=PATH; got {spec!r}")
    name, path = spec.split("=", 1)
    return name.strip(), path.strip()


def _first_by_hash(items: list[dict], n: int) -> list[dict]:
    return sorted(items, key=lambda it: hashlib.sha256(it["id"].encode()).hexdigest())[:n] if n else items


def cmd_export(args) -> None:
    sets: dict[str, list[dict]] = {}
    for spec in args.task:
        name, path = _named(spec, "--task")
        sets[name] = [item_from_task(name, t) for t in _read_jsonl(path)]
    if args.simple:
        sets["simple"] = [item_from_simple(r) for r in _read_jsonl(args.simple)]
    if args.massive:
        from sors.data.massive import load_massive
        items = items_from_labeled("massive", load_massive(partition="test"), instructions="What does the user want?")
        sets["massive"] = _first_by_hash(items, args.massive)
    if args.boolq:
        from sors.data.boolq import load_boolq
        sets["boolq"] = _first_by_hash(items_from_labeled("boolq", load_boolq()[1]), args.boolq)
    for spec in args.suite:
        name, path = _named(spec, "--suite")
        sets[name] = [item_from_suite_row(name, r) for r in _read_jsonl(path)]
    if not sets:
        raise SystemExit("nothing to export: give --task, --simple, --massive, --boolq or --suite")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, items in sets.items():
        if len({it["id"] for it in items}) != len(items):
            raise SystemExit(f"{name}: repeated ids")
        body = "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items)
        (out / f"{name}.jsonl").write_text(body, encoding="utf-8")
        manifest[name] = {"items": len(items), "questions": sum(len(it["gold"]) for it in items),
                          "sha256": hashlib.sha256(body.encode()).hexdigest()}
        print(f"{name:20} {len(items):6} items")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")


def _poster(base_url: str, model: str, timeout: float, api_key: str | None):
    url = base_url.rstrip("/") + "/v1/systemone"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key

    def post(request: dict) -> dict:
        body = json.dumps({"model": model, **request}).encode()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, body, headers), timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from None

    return post


def cmd_run(args) -> None:
    panel = pathlib.Path(args.panel)
    manifest = json.loads((panel / "manifest.json").read_text())
    names = args.sets.split(",") if args.sets else list(manifest)
    unknown = [n for n in names if n not in manifest]
    if unknown:
        raise SystemExit(f"--sets: not in the panel: {unknown}; it has {list(manifest)}")
    items = [it for n in names for it in _read_jsonl(panel / f"{n}.jsonl")]
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    post = _poster(args.base_url, args.model, args.timeout, args.api_key or os.environ.get("SORS_API_KEY"))
    print(f"{len(items)} items over {len(names)} sets -> {out}", flush=True)
    run_items(items, post, out, workers=args.workers)
    _print_table([out])


def _fmt(v) -> str:
    return f"{100 * v:6.1f}" if v is not None else f"{'-':>6}"


def _print_table(paths: list[pathlib.Path]) -> None:
    tables = [summarize(load_results(p)) for p in paths]
    names = list(dict.fromkeys(n for t in tables for n in t))
    cols = ("top1", "choice", "noul", "score")
    print(f"{'set':20}{'model':24}{'n':>6}{'err':>5}" + "".join(f"{c:>8}" for c in cols))
    for name in names:
        for p, t in zip(paths, tables):
            row = t.get(name)
            if row is None:
                continue
            print(f"{name:20}{p.stem[:23]:24}{row['n']:6}{row['errors']:5}" + "".join(f"  {_fmt(row[c])}" for c in cols))


def cmd_report(args) -> None:
    _print_table([pathlib.Path(p) for p in args.results])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="导出面板文件")
    ex.add_argument("--out", required=True)
    ex.add_argument("--task", action="append", default=[], help="NAME=PATH, JevBench 题目格式的 jsonl, 可重复")
    ex.add_argument("--simple", help="synth-simple-eval 的 jsonl")
    ex.add_argument("--massive", type=int, default=0, help="MASSIVE test 取几条 (0 = 不要)")
    ex.add_argument("--boolq", type=int, default=0, help="BoolQ validation 取几条 (0 = 不要)")
    ex.add_argument("--suite", action="append", default=[], help="NAME=PATH, Decision Index 题集的行文件 (.jsonl 或 .jsonl.gz)")
    ex.set_defaults(func=cmd_export)
    ru = sub.add_parser("run", help="对一个服务打分")
    ru.add_argument("--panel", required=True)
    ru.add_argument("--sets", help="逗号分隔, 默认面板里全部")
    ru.add_argument("--base-url", required=True)
    ru.add_argument("--model", required=True)
    ru.add_argument("--out", required=True)
    ru.add_argument("--workers", type=int, default=4)
    ru.add_argument("--timeout", type=float, default=600)
    ru.add_argument("--api-key", default=None, help="不给就读 SORS_API_KEY")
    ru.set_defaults(func=cmd_run)
    rp = sub.add_parser("report", help="出表, 几个结果文件并排")
    rp.add_argument("results", nargs="+")
    rp.set_defaults(func=cmd_report)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
