from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import write_dataframe, write_json


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def null_summary(df: pd.DataFrame) -> dict[str, int]:
    return df.isna().sum().sort_values(ascending=False).astype(int).to_dict()


def _parse_duration_minutes(value: str | float | None) -> float:
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if not text or text in {"unknown", "nan"}:
        return np.nan
    hours = re.search(r"(\d+)\s*hr", text)
    minutes = re.search(r"(\d+)\s*min", text)
    total = 0
    if hours:
        total += int(hours.group(1)) * 60
    if minutes:
        total += int(minutes.group(1))
    return float(total) if total > 0 else np.nan


def build_validation_report(
    df: pd.DataFrame,
    reports_dir: Path,
    rejected_path: Path,
    strict: bool = False,
) -> dict[str, str]:
    checks = pd.DataFrame(index=df.index)
    checks["missing_anime_id"] = df["anime_id"].isna() | (df["anime_id"].astype(str).str.strip() == "")
    checks["missing_name"] = df["name"].isna() | (df["name"].astype(str).str.strip() == "")

    numeric_rules = {
        "score": lambda s: s.notna() & ~s.between(0, 10, inclusive="both"),
        "episodes": lambda s: s.notna() & (s < 0),
        "members": lambda s: s.notna() & (s < 0),
        "favorites": lambda s: s.notna() & (s < 0),
        "scored_by": lambda s: s.notna() & (s < 0),
        "popularity": lambda s: s.notna() & (s <= 0),
        "rank": lambda s: s.notna() & (s <= 0),
    }
    for column, rule in numeric_rules.items():
        if column in df.columns:
            series = pd.to_numeric(df[column], errors="coerce")
            checks[f"invalid_{column}"] = rule(series)

    if "duration" in df.columns:
        duration_minutes = df["duration"].map(_parse_duration_minutes)
        checks["invalid_duration"] = duration_minutes.notna() & (duration_minutes <= 0)
    else:
        checks["invalid_duration"] = False

    checks["duplicate_anime_id"] = df.duplicated(subset=["anime_id"], keep=False)

    rejection_columns = [column for column in checks.columns if column in {"missing_anime_id", "missing_name"}]
    if strict:
        rejection_columns = list(checks.columns)

    rejected_mask = checks[rejection_columns].any(axis=1)
    rejected = df.loc[rejected_mask].copy()
    if not rejected.empty:
        rejected["rejection_reasons"] = checks.loc[rejected_mask].apply(
            lambda row: [column for column, flag in row.items() if bool(flag)],
            axis=1,
        )
    write_dataframe(rejected, rejected_path)

    report = {
        "row_count": int(len(df)),
        "strict_mode": strict,
        "check_failures": {column: int(checks[column].sum()) for column in checks.columns},
        "rejected_rows": int(rejected_mask.sum()),
        "warning_rows": int(checks.any(axis=1).sum()),
        "rejected_dataset_path": str(rejected_path),
    }
    report_path = reports_dir / "validation_report.json"
    write_json(report, report_path)
    return {"report": str(report_path), "rejected": str(rejected_path)}
