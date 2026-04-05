from __future__ import annotations

import logging
from pathlib import Path

from src.data import build_feature_matrix, clean, ingest, profile, transform


def run_stage(stage: str, data_config: dict, features_config: dict, interim_dir: Path, processed_dir: Path, checkpoints_dir: Path, reports_dir: Path, logger: logging.Logger, overwrite: bool) -> dict[str, str]:
    if stage == "ingest":
        return ingest.run(data_config, interim_dir, logger, overwrite=overwrite)
    if stage == "profile":
        return profile.run(interim_dir / "ingested_anime.parquet", reports_dir, logger, overwrite=overwrite)
    if stage == "clean":
        return clean.run(interim_dir / "ingested_anime.parquet", data_config, interim_dir, logger, overwrite=overwrite)
    if stage == "transform":
        return transform.run(interim_dir / "cleaned_anime.parquet", interim_dir, logger, overwrite=overwrite)
    if stage == "build_features":
        return build_feature_matrix.run(
            interim_dir / "transformed_anime.parquet",
            features_config,
            processed_dir,
            checkpoints_dir,
            logger,
            overwrite=overwrite,
        )
    return {}
