from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils.io import read_dataframe, write_json
from src.utils.validation import null_summary


NUMERIC_COLUMNS = ["score", "episodes", "rank", "popularity", "favorites", "scored_by", "members"]
CATEGORICAL_COLUMNS = ["type", "source", "rating", "status"]
MULTI_LABEL_COLUMNS = ["genres", "themes", "demographics"]


def _distribution_summary(series: pd.Series) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return {}
    return {
        "count": int(numeric.count()),
        "mean": float(numeric.mean()),
        "std": float(numeric.std()) if numeric.count() > 1 else 0.0,
        "min": float(numeric.min()),
        "median": float(numeric.median()),
        "max": float(numeric.max()),
    }


def run(ingested_path: Path, reports_dir: Path, logger: logging.Logger, overwrite: bool = False) -> dict[str, str]:
    json_path = reports_dir / "eda_summary.json"
    md_path = reports_dir / "eda_summary.md"
    if json_path.exists() and md_path.exists() and not overwrite:
        logger.info("Skipping profile; outputs already exist")
        return {"json": str(json_path), "markdown": str(md_path)}

    df = read_dataframe(ingested_path)
    missing = null_summary(df)
    duplicate_count = int(df.duplicated(subset=["anime_id"]).sum()) if "anime_id" in df.columns else 0

    summary = {
        "shape": {"rows": int(len(df)), "columns": int(df.shape[1])},
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_values": missing,
        "unique_counts": {column: int(df[column].nunique(dropna=True)) for column in df.columns},
        "duplicate_anime_id_count": duplicate_count,
        "numeric_summaries": {
            column: _distribution_summary(df[column]) for column in NUMERIC_COLUMNS if column in df.columns
        },
        "top_categories": {
            column: df[column].fillna("missing").astype(str).value_counts().head(10).to_dict()
            for column in CATEGORICAL_COLUMNS
            if column in df.columns
        },
        "multi_label_summaries": {
            column: (
                df[column]
                .fillna("")
                .astype(str)
                .str.split(",")
                .explode()
                .str.strip()
                .replace("", pd.NA)
                .dropna()
                .value_counts()
                .head(20)
                .to_dict()
            )
            for column in MULTI_LABEL_COLUMNS
            if column in df.columns
        },
        "high_missing_columns": [column for column, count in missing.items() if count / max(len(df), 1) > 0.3],
        "high_cardinality_columns": [
            column for column in df.columns if df[column].nunique(dropna=True) / max(len(df), 1) > 0.2
        ],
    }
    write_json(summary, json_path)

    useful_columns = ["score", "members", "favorites", "scored_by", "episodes", "duration", "type", "source", "rating", "genres"]
    markdown = [
        "# EDA Summary",
        f"- Rows: {len(df)}",
        f"- Columns: {df.shape[1]}",
        f"- Duplicate anime IDs: {duplicate_count}",
        f"- High-missing columns: {', '.join(summary['high_missing_columns']) if summary['high_missing_columns'] else 'None'}",
        f"- High-cardinality columns: {', '.join(summary['high_cardinality_columns'][:10]) if summary['high_cardinality_columns'] else 'None'}",
        "",
        "## Clustering-Relevant Columns",
        ", ".join([column for column in useful_columns if column in df.columns]),
        "",
        "## Notes",
        "- Numeric popularity, engagement, and rank columns look useful after careful scaling/log transforms.",
        "- Multi-label genres should be explicitly vocabulary-controlled before encoding.",
        "- Identifier and title columns should be preserved outside the model matrix for interpretability.",
    ]
    md_path.write_text("\n".join(markdown), encoding="utf-8")
    logger.info("Saved EDA summaries to %s and %s", json_path, md_path)
    return {"json": str(json_path), "markdown": str(md_path)}
