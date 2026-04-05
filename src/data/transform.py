from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import read_dataframe, write_dataframe, write_json


def _premiered_season(value: object) -> str:
    if value is None or pd.isna(value):
        return "missing"
    text = str(value).strip().lower()
    if text in {"unknown", "missing", ""}:
        return "missing"
    return text.split()[0]


def _parse_multilabel_column(series: pd.Series) -> pd.Series:
    return series.fillna("missing").astype(str).map(
        lambda value: [item.strip() for item in value.split("|") if item.strip() and item.strip() != "missing"]
    )


def run(
    cleaned_path: Path,
    features_config: dict,
    processed_dir: Path,
    checkpoints_dir: Path,
    logger: logging.Logger,
    overwrite: bool = False,
) -> dict[str, str]:
    output_path = processed_dir / "anime_transformed.parquet"
    feature_metadata_path = checkpoints_dir / "feature_metadata.json"
    encoder_metadata_path = checkpoints_dir / "encoder_metadata.json"
    if output_path.exists() and feature_metadata_path.exists() and encoder_metadata_path.exists() and not overwrite:
        logger.info("Skipping transform; outputs already exist")
        return {
            "dataset": str(output_path),
            "feature_metadata": str(feature_metadata_path),
            "encoder_metadata": str(encoder_metadata_path),
        }

    df = read_dataframe(cleaned_path).copy()
    df["premiered_season"] = df["premiered"].map(_premiered_season) if "premiered" in df.columns else "missing"

    numeric_columns = [column for column in features_config["numeric_columns"] if column in df.columns]
    categorical_columns = [column for column in features_config["categorical_columns"] if column in df.columns]
    multi_label_columns = [column for column in features_config["multi_label_columns"] if column in df.columns]

    transformed_numeric_columns = []
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in features_config.get("log_transform_columns", []):
        if column in df.columns:
            new_column = f"{column}_log1p"
            df[new_column] = np.log1p(df[column].clip(lower=0))
            transformed_numeric_columns.append(new_column)

    for column in categorical_columns:
        df[column] = df[column].fillna("missing").astype(str).str.strip().str.lower()

    vocabularies: dict[str, list[str]] = {}
    for column in multi_label_columns:
        tags = _parse_multilabel_column(df[column])
        counts = tags.explode().value_counts()
        vocab = [tag for tag in counts.index.tolist() if tag][:50]
        vocabularies[column] = vocab
        for tag in vocab:
            df[f"{column}__{tag}"] = tags.map(lambda items: int(tag in items))

    write_dataframe(df, output_path)
    write_json(
        {
            "identifier_columns": [column for column in ["anime_id", "name"] if column in df.columns],
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "multi_label_columns": multi_label_columns,
            "log_transformed_columns": transformed_numeric_columns,
            "final_candidate_feature_columns": [
                column
                for column in df.columns
                if column not in {"anime_id", "name", "english_name", "other_name", "synopsis", "image_url"}
            ],
        },
        feature_metadata_path,
    )
    write_json(
        {
            "categorical_columns": categorical_columns,
            "multi_label_vocabularies": vocabularies,
            "top_retained_tags": vocabularies,
        },
        encoder_metadata_path,
    )
    logger.info("Saved transformed dataset to %s", output_path)
    return {
        "dataset": str(output_path),
        "feature_metadata": str(feature_metadata_path),
        "encoder_metadata": str(encoder_metadata_path),
    }
