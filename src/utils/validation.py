from __future__ import annotations

from pathlib import Path

import pandas as pd


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def null_summary(df: pd.DataFrame) -> dict[str, int]:
    return df.isna().sum().sort_values(ascending=False).astype(int).to_dict()


def stage_should_skip(output_path: Path, skip_existing: bool) -> bool:
    return skip_existing and output_path.exists()
