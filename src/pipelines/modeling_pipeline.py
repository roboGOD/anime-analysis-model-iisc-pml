from __future__ import annotations

import logging
from pathlib import Path

from src.clustering.base import ClusteringModelAdapter
from src.modeling import diagnostics, evaluate
from src.modeling.artifacts import matrix_npy_path, model_assignments_parquet_path
from src.modeling.assignments import run as assign_clusters
from src.modeling.reduce_features import run as reduce_features
from src.modeling.select_model import run as select_model
from src.modeling.train_model import run as train_model


def run_stage(
    stage: str,
    adapter: ClusteringModelAdapter,
    model_config: dict,
    base_config: dict,
    checkpoints_dir: Path,
    models_dir: Path,
    metrics_dir: Path,
    plots_dir: Path,
    processed_dir: Path,
    reports_dir: Path,
    logger: logging.Logger,
    overwrite: bool,
) -> dict[str, str]:
    matrix_path = matrix_npy_path(processed_dir, model_config, adapter.model_name)
    if stage == "select_model":
        return select_model(
            matrix_path,
            adapter,
            model_config,
            base_config,
            metrics_dir,
            reports_dir,
            logger,
            overwrite=overwrite,
        )
    if stage == "train_model":
        return train_model(
            matrix_path,
            adapter,
            model_config,
            base_config,
            models_dir,
            metrics_dir,
            reports_dir,
            checkpoints_dir,
            logger,
            overwrite=overwrite,
        )
    if stage == "assign_clusters":
        return assign_clusters(
            checkpoints_dir / "row_mapping.parquet",
            matrix_path,
            models_dir / f"final_{adapter.model_name}.joblib",
            adapter,
            reports_dir,
            logger,
            overwrite=overwrite,
        )
    if stage == "reduce_features":
        return reduce_features(
            processed_dir,
            checkpoints_dir,
            reports_dir,
            plots_dir,
            model_config,
            logger,
            overwrite=overwrite,
        )
    if stage == "evaluate":
        return evaluate.run(
            adapter.model_name,
            model_assignments_parquet_path(reports_dir, adapter.model_name),
            matrix_path,
            metrics_dir / f"{adapter.model_name}_model_selection.json",
            processed_dir / "anime_cleaned.parquet",
            metrics_dir,
            reports_dir,
            logger,
            overwrite=overwrite,
        )
    if stage == "diagnostics":
        return diagnostics.run(
            models_dir / f"{adapter.model_name}_model_metadata.json",
            model_assignments_parquet_path(reports_dir, adapter.model_name),
            reports_dir,
            logger,
            overwrite=overwrite,
        )
    return {}
