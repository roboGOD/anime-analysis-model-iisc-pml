from __future__ import annotations

import argparse
from pathlib import Path

from src.data import build_feature_matrix, clean, ingest, profile, transform
from src.utils.config import load_project_configs, snapshot_configs
from src.utils.io import read_dataframe
from src.utils.logging_utils import setup_logging
from src.utils.paths import build_run_id, get_run_paths, resolve_path
from src.utils.reproducibility import get_git_commit_hash, set_global_seed
from src.utils.validation import build_validation_report
from src.visualization.eda_plots import run_profile_plots, run_transformed_plots


STAGES = ["ingest", "profile", "validate", "clean", "transform", "build_matrix", "all"]


def run_data_pipeline(
    config_dir: Path,
    stage: str = "all",
    run_id: str | None = None,
    overwrite: bool = False,
    skip_existing: bool = False,
) -> dict[str, str]:
    configs = load_project_configs(config_dir)
    base_config = configs["base"]
    data_config = configs["data"]
    features_config = configs["features"]

    run_id = build_run_id(run_id)
    run_paths = get_run_paths(base_config, run_id)
    logger = setup_logging(run_paths.log_file, level=base_config["runtime"]["logging_level"])
    set_global_seed(int(base_config["runtime"]["random_seed"]))
    logger.info("Data pipeline start | run_id=%s | git=%s", run_id, get_git_commit_hash())
    snapshot_configs(configs, run_paths.run_configs_dir / f"{run_id}_data_pipeline.json")

    effective_overwrite = overwrite or base_config["runtime"]["overwrite"]
    if skip_existing:
        effective_overwrite = False

    interim_dir = resolve_path(base_config["paths"]["interim_data_dir"])
    processed_dir = resolve_path(base_config["paths"]["processed_data_dir"])
    selected = STAGES[:-1] if stage == "all" else [stage]
    artifacts: dict[str, str] = {}

    if "ingest" in selected:
        artifacts.update(ingest.run(data_config, interim_dir, logger, overwrite=effective_overwrite))
    if "profile" in selected:
        ingested_path = interim_dir / "anime_ingested.parquet"
        artifacts.update(profile.run(ingested_path, run_paths.reports_dir, logger, overwrite=effective_overwrite))
        artifacts.update(run_profile_plots(ingested_path, run_paths.plots_dir, logger, overwrite=effective_overwrite))
    if "validate" in selected:
        artifacts.update(
            build_validation_report(
                read_dataframe(interim_dir / "anime_ingested.parquet"),
                run_paths.reports_dir,
                interim_dir / "anime_rejected.parquet",
                strict=False,
            )
        )
    if "clean" in selected:
        artifacts.update(
            clean.run(
                interim_dir / "anime_ingested.parquet",
                data_config,
                processed_dir,
                run_paths.reports_dir,
                logger,
                overwrite=effective_overwrite,
            )
        )
    if "transform" in selected:
        artifacts.update(
            transform.run(
                processed_dir / "anime_cleaned.parquet",
                features_config,
                processed_dir,
                run_paths.checkpoints_dir,
                logger,
                overwrite=effective_overwrite,
            )
        )
    if "build_matrix" in selected:
        artifacts.update(
            build_feature_matrix.run(
                processed_dir / "anime_transformed.parquet",
                features_config,
                processed_dir,
                run_paths.checkpoints_dir,
                run_paths.metrics_dir,
                run_paths.reports_dir,
                logger,
                overwrite=effective_overwrite,
            )
        )
        artifacts.update(
            run_transformed_plots(
                processed_dir / "anime_transformed.parquet",
                run_paths.plots_dir,
                logger,
                overwrite=effective_overwrite,
            )
        )
    logger.info("Data pipeline complete | stage=%s", stage)
    return artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the anime data pipeline")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_data_pipeline(
        config_dir=resolve_path(args.config_dir),
        stage=args.stage,
        run_id=args.run_id,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
