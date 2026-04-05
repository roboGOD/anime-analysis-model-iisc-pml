from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from src.utils.io import read_dataframe, write_dataframe, write_json


NUMERIC_COLUMNS = ["score", "episodes", "rank", "popularity", "favorites", "scored_by", "members"]
CATEGORICAL_COLUMNS = ["type", "source", "rating", "status", "premiered"]
MULTI_LABEL_COLUMNS = ["genres", "themes", "demographics"]


def _clean_label_text(value: object) -> object:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return pd.NA
    return re.sub(r"\s+", " ", text)


def _normalize_multilabel(value: object) -> str:
    if value is None or pd.isna(value):
        return "missing"
    tags = [re.sub(r"\s+", " ", part.strip().lower()) for part in str(value).split(",")]
    tags = sorted({tag for tag in tags if tag and tag != "unknown"})
    return "|".join(tags) if tags else "missing"


def _parse_duration_minutes(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    hours = re.search(r"(\d+)\s*hr", text)
    minutes = re.search(r"(\d+)\s*min", text)
    total = 0
    if hours:
        total += int(hours.group(1)) * 60
    if minutes:
        total += int(minutes.group(1))
    return float(total) if total > 0 else None


def run(
    ingested_path: Path,
    data_config: dict,
    processed_dir: Path,
    reports_dir: Path,
    logger: logging.Logger,
    overwrite: bool = False,
) -> dict[str, str]:
    output_path = processed_dir / "anime_cleaned.parquet"
    summary_path = reports_dir / "cleaning_summary.json"
    if output_path.exists() and summary_path.exists() and not overwrite:
        logger.info("Skipping clean; outputs already exist")
        return {"dataset": str(output_path), "summary": str(summary_path)}

    df = read_dataframe(ingested_path)
    original_rows = len(df)
    missing_tokens = {str(token).strip().lower() for token in data_config.get("missing_value_tokens", []) if token is not None}

    replacements_applied: dict[str, int] = {}
    for column in df.columns:
        if df[column].dtype == "object":
            before_missing = int(df[column].isna().sum())
            df[column] = df[column].map(_clean_label_text)
            df[column] = df[column].map(
                lambda value: pd.NA
                if isinstance(value, str) and value.strip().lower() in missing_tokens
                else value
            )
            after_missing = int(df[column].isna().sum())
            replacements_applied[column] = max(after_missing - before_missing, 0)

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["duration_minutes"] = df["duration"].map(_parse_duration_minutes) if "duration" in df.columns else pd.NA

    before_dedup = len(df)
    df = df.drop_duplicates(
        subset=data_config["deduplication"]["subset"],
        keep=data_config["deduplication"].get("keep", "first"),
    )
    duplicates_removed = before_dedup - len(df)

    if "anime_id" in df.columns:
        df = df[df["anime_id"].notna()]

    for column in CATEGORICAL_COLUMNS:
        if column in df.columns:
            df[column] = df[column].fillna("missing").astype(str).str.strip().str.lower()

    for column in MULTI_LABEL_COLUMNS:
        if column in df.columns:
            df[column] = df[column].map(_normalize_multilabel)

    imputations: dict[str, str] = {}
    for column in NUMERIC_COLUMNS + ["duration_minutes"]:
        if column in df.columns:
            if df[column].notna().any():
                fill_value = float(df[column].median())
                df[column] = df[column].fillna(fill_value)
                imputations[column] = f"median:{fill_value}"

    rows_with_any_imputation = int(
        sum(1 for value in replacements_applied.values() if value > 0)
    )

    write_dataframe(df, output_path)
    write_json(
        {
            "original_rows": int(original_rows),
            "cleaned_rows": int(len(df)),
            "rows_dropped": int(original_rows - len(df)),
            "duplicates_removed": int(duplicates_removed),
            "missing_token_replacements": replacements_applied,
            "imputations": imputations,
            "rows_with_any_imputation_applied_estimate": rows_with_any_imputation,
        },
        summary_path,
    )
    logger.info("Saved cleaned dataset to %s with %s rows", output_path, len(df))
    return {"dataset": str(output_path), "summary": str(summary_path)}
