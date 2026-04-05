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


def run(data_config: dict, interim_dir: Path, logger: logging.Logger, overwrite: bool = False) -> dict[str, str]:
    raw_path = Path(data_config["raw_files"]["primary"])
    output_path = interim_dir / "ingested_anime.parquet"
    schema_path = interim_dir / "ingested_schema.json"

    if output_path.exists() and not overwrite:
        logger.info("Skipping ingest; output already exists at %s", output_path)
        return {"dataset": str(output_path), "schema": str(schema_path)}

    logger.info("Starting ingest from %s", raw_path)
    df = read_dataframe(raw_path)
    validate_columns(df, data_config["required_columns"])

    standardized_columns = {_snake_case(column): column for column in df.columns}
    df.columns = list(standardized_columns.keys())
    logger.info("Ingested dataframe shape=%s", df.shape)

    write_dataframe(df, output_path)
    write_json(
        {
            "rows": int(df.shape[0]),
            "columns": list(df.columns),
            "original_columns": standardized_columns,
        },
        schema_path,
    )
    logger.info("Saved ingested dataset to %s", output_path)
    return {"dataset": str(output_path), "schema": str(schema_path)}
