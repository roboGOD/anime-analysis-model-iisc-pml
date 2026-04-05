from __future__ import annotations

import logging
from pathlib import Path

from src.clustering.base import ClusteringModelAdapter
from src.visualization import eda_plots, model_plots, report_plots


def run_stage(
    stage: str,
    adapter: ClusteringModelAdapter,
    interim_dir: Path,
    processed_dir: Path,
    checkpoints_dir: Path,
    metrics_dir: Path,
    plots_dir: Path,
    logger: logging.Logger,
    overwrite: bool,
) -> dict[str, str]:
    if stage == "plot_eda":
        return eda_plots.run_profile_plots(interim_dir / "anime_ingested.parquet", plots_dir, logger, overwrite=overwrite)
    if stage == "plot_model":
        return model_plots.run(
            metrics_dir / f"{adapter.model_name}_model_selection.csv",
            metrics_dir / "gmm_cluster_sizes.csv",
            plots_dir,
            logger,
            overwrite=overwrite,
        )
    if stage == "plot_report":
        return report_plots.run(
            reports_dir / "cluster_assignments.parquet",
            processed_dir / "X_gmm.npy",
            processed_dir / "anime_cleaned.parquet",
            plots_dir,
            logger,
            overwrite=overwrite,
        )
    return {}
