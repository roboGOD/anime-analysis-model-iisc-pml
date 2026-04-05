from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from src.utils.io import read_dataframe, write_dataframe, write_json
from src.utils.validation import validate_columns


def _snake_case(name: str) -> str:
    value = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip())
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return value.strip("_").lower()


def _normalize_text(value: object) -> object:
    if not isinstance(value, str):
        return value
    normalized = " ".join(value.replace("\r\n", "\n").replace("\r", "\n").split())
    return normalized.strip()


def run(data_config: dict, interim_dir: Path, logger: logging.Logger, overwrite: bool = False) -> dict[str, str]:
    raw_path = Path(data_config["raw_files"]["primary"])
    output_path = interim_dir / "anime_ingested.parquet"
    schema_path = interim_dir / "anime_ingested_schema.json"

    if output_path.exists() and schema_path.exists() and not overwrite:
        logger.info("Skipping ingest; outputs already exist at %s", output_path)
        return {"dataset": str(output_path), "schema": str(schema_path)}

    df = read_dataframe(raw_path)
    validate_columns(df, data_config["required_columns"])

    original_columns = list(df.columns)
    df.columns = [_snake_case(column) for column in df.columns]
    object_columns = df.select_dtypes(include="object").columns
    for column in object_columns:
        df[column] = df[column].map(_normalize_text)

    write_dataframe(df, output_path)
    write_json(
        {
            "raw_path": str(raw_path),
            "row_count": int(len(df)),
            "column_count": int(df.shape[1]),
            "columns": list(df.columns),
            "column_mapping": dict(zip(df.columns, original_columns, strict=False)),
        },
        schema_path,
    )
    logger.info("Ingested %s rows and %s columns from %s", len(df), df.shape[1], raw_path)
    return {"dataset": str(output_path), "schema": str(schema_path)}
