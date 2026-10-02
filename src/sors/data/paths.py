"""Locate the trusted, separately versioned decidophobia-dataset checkout."""
from __future__ import annotations

import os
import pathlib

ENV = "SORS_DATASETS_DIR"
LEGACY_ENV = "DECIDOPHOBIA_DATASETS_DIR"
DEFAULT_ROOT = pathlib.Path(__file__).resolve().parents[4] / "decidophobia-dataset"


def datasets_root(value=None) -> pathlib.Path:
    """Explicit CLI/API directory, then environment, then the sibling checkout."""
    value = value if value is not None else os.environ.get(ENV, os.environ.get(LEGACY_ENV, DEFAULT_ROOT))
    if not str(value).strip():
        raise ValueError(f"--datasets-dir / {ENV} must name a dataset checkout")
    return pathlib.Path(value).expanduser().resolve()


def asset_path(name: str, datasets_dir=None) -> pathlib.Path:
    path = datasets_root(datasets_dir) / name
    if not path.exists():
        raise FileNotFoundError(f"dataset asset missing: {path}; set --datasets-dir or {ENV} "
                                "to the decidophobia-dataset repository root")
    return path


def add_datasets_argument(parser) -> None:
    parser.add_argument("--datasets-dir", default=None,
                        help=f"trusted decidophobia-dataset checkout root (default: {ENV}, then sibling checkout); "
                             "download caches remain separate")
