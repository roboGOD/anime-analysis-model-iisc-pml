from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def exists(path: Path) -> bool:
    return path.exists()


def read_dataframe(path: Path, **kwargs: Any) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    raise ValueError(f"Unsupported file format: {path}")


def _atomic_text_write(path: Path, content: str) -> None:
    ensure_parent(path)
    with NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def write_dataframe(df: pd.DataFrame, path: Path, index: bool = False, **kwargs: Any) -> None:
    ensure_parent(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=index, **kwargs)
        return
    if suffix == ".parquet":
        df.to_parquet(path, index=index, **kwargs)
        return
    raise ValueError(f"Unsupported file format: {path}")


def write_json(payload: Any, path: Path, indent: int = 2) -> None:
    _atomic_text_write(path, json.dumps(payload, indent=indent, ensure_ascii=False, default=str))


def versioned_path(directory: Path, stem: str, suffix: str, run_id: str) -> Path:
    return directory / f"{stem}_{run_id}{suffix}"
