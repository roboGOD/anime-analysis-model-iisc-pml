from __future__ import annotations

import logging
from pathlib import Path

from src.utils.io import read_dataframe, write_json
from src.utils.validation import null_summary


def run(ingested_path: Path, reports_dir: Path, logger: logging.Logger, overwrite: bool = False) -> dict[str, str]:
    output_path = reports_dir / "profile_summary.json"
    if output_path.exists() and not overwrite:
        logger.info("Skipping profile; output already exists at %s", output_path)
        return {"profile": str(output_path)}

    df = read_dataframe(ingested_path)
    summary = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "null_counts": null_summary(df),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "sample_unique_counts": {column: int(df[column].nunique(dropna=True)) for column in df.columns[:10]},
    }
    write_json(summary, output_path)
    logger.info("Saved profiling summary to %s", output_path)
    return {"profile": str(output_path)}
