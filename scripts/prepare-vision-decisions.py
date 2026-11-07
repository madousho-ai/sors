"""Prepare the local vision decision files and print their manifest entries (offline).

Expects the upstream parquet files downloaded with `hf download` under --root (default
data/public-decisions/vision):

  aokvqa/       HuggingFaceM4/A-OKVQA        data/train-*
  scienceqa/    derek-thomas/ScienceQA       data/train-*
  cauldron/     HuggingFaceM4/the_cauldron   ai2d/* iconqa/* vsr/* nlvr2/*
  mm-mind2web/  osunlp/Multimodal-Mind2Web   data/train-*

Two sources are rewritten into --root/prepared/ because reading them whole is too large:
Multimodal-Mind2Web drops its HTML and keeps the top of each screenshot
(public_vision.crop_mind2web_row), and NLVR2 keeps --nlvr2 questions drawn uniformly over all
its turns. The others are read as released. The entries (paths relative to the manifest
directory, sha256 of every file read and of every upstream file a prepared file came from)
go to --root/prepared/entries.json; merge them into the manifest's sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from sors.data.public_vision import (MIND2WEB_COLUMNS, MIND2WEB_VIEWPORT, crop_mind2web_row,  # noqa: E402
                                     sample_cauldron_questions)

REVISIONS = {
    "aokvqa": ("HuggingFaceM4/A-OKVQA", "d1b0efa3a436e9101dfbde3752db7607da696c35", "unspecified on the Hub card; "
               "A-OKVQA annotations Apache-2.0, COCO images under their Flickr terms"),
    "scienceqa": ("derek-thomas/ScienceQA", "f18b0a70359ebfb41f658fd564208d0355b013f4",
                  "CC-BY-SA-4.0 on the Hub card; original release CC BY-NC-SA 4.0"),
    "cauldron": ("HuggingFaceM4/the_cauldron", "847a98a779b1652d65111daf20c972dfcd333605",
                 "per source dataset; see the_cauldron card"),
    "mm-mind2web": ("osunlp/Multimodal-Mind2Web", "1b4c6a8cf9f77b7a5e0d641959935c80c4a05889", "OpenRAIL"),
}


def sha256(path: pathlib.Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def upstream(key: str, relative: str) -> str:
    repo, revision, _ = REVISIONS[key]
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{relative}"


def entry(dataset, key, path: pathlib.Path, manifest_dir: pathlib.Path, root: pathlib.Path, **extra):
    return {"dataset": dataset, "split": "train", "path": str(path.relative_to(manifest_dir)),
            "source": upstream(key, str(path.relative_to(root / key))),
            "license": REVISIONS[key][2], "sha256": sha256(path), **extra}


def write(rows, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def prepare_mind2web(root, out_dir, manifest_dir, height):
    entries = []
    for shard in sorted((root / "mm-mind2web/data").glob("train-*.parquet")):
        target = out_dir / "mm-mind2web" / shard.name
        if not target.exists():
            f = pq.ParquetFile(shard)
            rows = [crop_mind2web_row(r, height) for b in f.iter_batches(batch_size=32, columns=MIND2WEB_COLUMNS)
                    for r in b.to_pylist()]
            write(rows, target)
        print(f"mm-mind2web {shard.name} -> {target}", flush=True)
        entries.append({"dataset": "mm-mind2web", "split": "train", "path": str(target.relative_to(manifest_dir)),
                        "source": upstream("mm-mind2web", f"data/{shard.name}"),
                        "license": REVISIONS["mm-mind2web"][2], "sha256": sha256(target),
                        "prepared_from": {"path": str(shard.relative_to(manifest_dir)), "sha256": sha256(shard)},
                        "prepare": f"crop_mind2web_row(height={height}); HTML columns dropped"})
    return entries


def prepare_nlvr2(root, out_dir, manifest_dir, n, seed):
    shards = sorted((root / "cauldron/nlvr2").glob("train-*.parquet"))
    index = []  # one entry per upstream row, in shard order: its texts and its position
    for shard in shards:
        for b in pq.ParquetFile(shard).iter_batches(batch_size=512, columns=["texts"]):
            for r in b.to_pylist():
                index.append({"texts": r["texts"], "images": [], "_at": len(index)})
    picked = sample_cauldron_questions(index, n, seed)
    wanted = {r["_at"]: r["texts"] for r in picked}
    rows, seen = [], 0
    for shard in shards:
        for b in pq.ParquetFile(shard).iter_batches(batch_size=128, columns=["images"]):
            for r in b.to_pylist():
                if seen in wanted:
                    rows.append({"images": r["images"], "texts": wanted[seen]})
                seen += 1
    target = out_dir / "nlvr2" / f"train-sample{n}-seed{seed}.parquet"
    write(rows, target)
    print(f"nlvr2 {len(rows)} rows, {sum(len(r['texts']) for r in rows)} questions -> {target}", flush=True)
    return [{"dataset": "nlvr2", "split": "train", "path": str(target.relative_to(manifest_dir)),
             "source": upstream("cauldron", "nlvr2/"), "license": "CC-BY-4.0 (NLVR2)", "sha256": sha256(target),
             "prepared_from": [{"path": str(p.relative_to(manifest_dir)), "sha256": sha256(p)} for p in shards],
             "prepare": f"sample_cauldron_questions(n={n}, seed={seed})"}]


def main() -> None:
    here = pathlib.Path(__file__).resolve().parents[1] / "data/public-decisions"
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(here / "vision"))
    ap.add_argument("--manifest-dir", default=str(here), help="entry paths are written relative to this directory")
    ap.add_argument("--nlvr2", type=int, default=20000, help="NLVR2 questions to keep")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--viewport", type=int, default=MIND2WEB_VIEWPORT, help="screenshot height kept, pixels")
    args = ap.parse_args()
    root, manifest_dir = pathlib.Path(args.root).resolve(), pathlib.Path(args.manifest_dir).resolve()
    out_dir = root / "prepared"
    entries = []
    for dataset, key, pattern in [("aokvqa", "aokvqa", "data/train-*.parquet"),
                                  ("scienceqa", "scienceqa", "data/train-*.parquet"),
                                  ("ai2d", "cauldron", "ai2d/train-*.parquet"),
                                  ("iconqa", "cauldron", "iconqa/train-*.parquet"),
                                  ("vsr", "cauldron", "vsr/train-*.parquet")]:
        for path in sorted((root / key).glob(pattern)):
            entries.append(entry(dataset, key, path, manifest_dir, root))
    entries += prepare_nlvr2(root, out_dir, manifest_dir, args.nlvr2, args.seed)
    entries += prepare_mind2web(root, out_dir, manifest_dir, args.viewport)
    (out_dir / "entries.json").write_text(json.dumps(entries, indent=2))
    print(f"{len(entries)} entries -> {out_dir / 'entries.json'}")


if __name__ == "__main__":
    main()
