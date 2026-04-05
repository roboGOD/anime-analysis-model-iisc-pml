from __future__ import annotations

import logging
from pathlib import Path

from src.clustering.base import ClusteringModelAdapter
from src.modeling.artifacts import matrix_npy_path, model_assignments_parquet_path
from src.visualization import eda_plots, model_plots, report_plots


def run_stage(
    stage: str,
    adapter: ClusteringModelAdapter,
    model_config: dict,
    interim_dir: Path,
    processed_dir: Path,
    checkpoints_dir: Path,
    metrics_dir: Path,
    reports_dir: Path,
    plots_dir: Path,
    logger: logging.Logger,
    overwrite: bool,
) -> dict[str, str]:
    if stage == "plot_eda":
        return eda_plots.run_profile_plots(interim_dir / "anime_ingested.parquet", plots_dir, logger, overwrite=overwrite)
    if stage == "plot_model":
        return model_plots.run(
            adapter.model_name,
            metrics_dir / f"{adapter.model_name}_model_selection.csv",
            metrics_dir / f"{adapter.model_name}_cluster_sizes.csv",
            plots_dir,
            logger,
            overwrite=overwrite,
        )
    if stage == "plot_report":
        return report_plots.run(
            adapter.model_name,
            model_assignments_parquet_path(reports_dir, adapter.model_name),
            matrix_npy_path(processed_dir, model_config, adapter.model_name),
            processed_dir / "anime_cleaned.parquet",
            plots_dir,
            logger,
            overwrite=overwrite,
        )
    return {}
