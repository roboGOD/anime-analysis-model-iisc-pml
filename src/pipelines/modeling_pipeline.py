from __future__ import annotations

import logging
from pathlib import Path

from src.clustering.base import ClusteringModelAdapter
from src.modeling import diagnostics, evaluate
from src.modeling.assignments import run as assign_clusters
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
    processed_dir: Path,
    reports_dir: Path,
    logger: logging.Logger,
    overwrite: bool,
) -> dict[str, str]:
    if stage == "select_model":
        return select_model(
            processed_dir / "X_gmm.npy",
            adapter,
            model_config,
            base_config,
            metrics_dir,
            logger,
            overwrite=overwrite,
        )
    if stage == "train_model":
        return train_model(
            processed_dir / "X_gmm.npy",
            adapter,
            model_config,
            base_config,
            models_dir,
            metrics_dir,
            logger,
            overwrite=overwrite,
        )
    if stage == "assign_clusters":
        return assign_clusters(
            checkpoints_dir / "row_mapping.parquet",
            processed_dir / "X_gmm.npy",
            models_dir / f"{adapter.model_name}_model.joblib",
            adapter,
            processed_dir,
            logger,
            overwrite=overwrite,
        )
    if stage == "evaluate":
        return evaluate.run(
            processed_dir / f"{adapter.model_name}_cluster_assignments.parquet",
            metrics_dir,
            logger,
            overwrite=overwrite,
        )
    if stage == "diagnostics":
        return diagnostics.run(
            models_dir / f"{adapter.model_name}_model_metadata.json",
            reports_dir,
            logger,
            overwrite=overwrite,
        )
    return {}
