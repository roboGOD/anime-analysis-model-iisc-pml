from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from src.utils.io import read_dataframe, write_dataframe, write_json


def _parse_duration_minutes(value: str | float | None) -> float:
    if value is None or value != value:
        return np.nan
    text = str(value).lower()
    hours = re.search(r"(\d+)\s*hr", text)
    minutes = re.search(r"(\d+)\s*min", text)
    total = 0
    if hours:
        total += int(hours.group(1)) * 60
    if minutes:
        total += int(minutes.group(1))
    if "per ep" in text and total:
        return float(total)
    if total:
        return float(total)
    return np.nan


def _parse_premiered_season(value: str | float | None) -> str:
    if value is None or value != value:
        return "unknown"
    text = str(value).strip().lower()
    return text.split()[0] if text and text != "unknown" else "unknown"


def run(cleaned_path: Path, interim_dir: Path, logger: logging.Logger, overwrite: bool = False) -> dict[str, str]:
    output_path = interim_dir / "transformed_anime.parquet"
    metadata_path = interim_dir / "transformation_metadata.json"
    if output_path.exists() and not overwrite:
        logger.info("Skipping transform; output already exists at %s", output_path)
        return {"dataset": str(output_path), "metadata": str(metadata_path)}

    df = read_dataframe(cleaned_path)
    df["genres"] = df["genres"].fillna("").map(lambda value: [part.strip() for part in str(value).split(",") if part.strip()])
    df["duration_minutes"] = df["duration"].map(_parse_duration_minutes) if "duration" in df.columns else np.nan
    df["premiered_season"] = df["premiered"].map(_parse_premiered_season) if "premiered" in df.columns else "unknown"

    for column in ["favorites", "scored_by", "members", "popularity"]:
        if column in df.columns:
            df[f"log1p_{column}"] = np.log1p(df[column].clip(lower=0))

    write_dataframe(df, output_path)
    write_json(
        {
            "added_columns": [
                "genres",
                "duration_minutes",
                "premiered_season",
                "log1p_favorites",
                "log1p_scored_by",
                "log1p_members",
                "log1p_popularity",
            ]
        },
        metadata_path,
    )
    logger.info("Saved transformed dataset to %s", output_path)
    return {"dataset": str(output_path), "metadata": str(metadata_path)}
