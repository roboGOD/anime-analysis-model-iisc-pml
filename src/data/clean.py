from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils.io import read_dataframe, write_dataframe, write_json


NUMERIC_COLUMNS = ["score", "episodes", "rank", "popularity", "favorites", "scored_by", "members"]


def run(
    ingested_path: Path,
    data_config: dict,
    interim_dir: Path,
    logger: logging.Logger,
    overwrite: bool = False,
) -> dict[str, str]:
    output_path = interim_dir / "cleaned_anime.parquet"
    report_path = interim_dir / "cleaning_report.json"
    if output_path.exists() and not overwrite:
        logger.info("Skipping clean; output already exists at %s", output_path)
        return {"dataset": str(output_path), "report": str(report_path)}

    df = read_dataframe(ingested_path)
    original_rows = len(df)

    missing_tokens = {token for token in data_config.get("missing_value_tokens", []) if token is not None}
    df = df.replace(list(missing_tokens), pd.NA)

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    drop_reasons: dict[str, int] = {}

    before_dedup = len(df)
    df = df.drop_duplicates(
        subset=[column.lower() for column in data_config["deduplication"]["subset"]],
        keep=data_config["deduplication"].get("keep", "first"),
    )
    drop_reasons["duplicates"] = before_dedup - len(df)

    filters = data_config["filters"]
    before = len(df)
    df = df[df["score"].between(filters["min_score"], filters["max_score"], inclusive="both") | df["score"].isna()]
    drop_reasons["invalid_score"] = before - len(df)

    before = len(df)
    df = df[(df["members"] >= filters["min_members"]) | df["members"].isna()]
    drop_reasons["low_members"] = before - len(df)

    before = len(df)
    df = df[df["status"].isin(filters["allowed_status"]) | df["status"].isna()]
    drop_reasons["invalid_status"] = before - len(df)

    write_dataframe(df, output_path)
    write_json(
        {
            "original_rows": original_rows,
            "final_rows": len(df),
            "dropped_rows": original_rows - len(df),
            "drop_reasons": drop_reasons,
            "null_counts": df.isna().sum().astype(int).to_dict(),
        },
        report_path,
    )
    logger.info("Saved cleaned dataset to %s with rows=%s dropped=%s", output_path, len(df), original_rows - len(df))
    return {"dataset": str(output_path), "report": str(report_path)}
