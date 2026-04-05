from __future__ import annotations

import argparse
from pathlib import Path

from src.clustering.registry import create_model_adapter
from src.utils.config import load_project_configs, snapshot_configs
from src.utils.logging_utils import setup_logging
from src.utils.paths import build_run_id, get_run_paths, resolve_path
from src.utils.reproducibility import get_git_commit_hash, set_global_seed
from src.pipelines import data_pipeline, modeling_pipeline, reporting_pipeline


AVAILABLE_STAGES = [
    "ingest",
    "profile",
    "validate",
    "clean",
    "transform",
    "build_matrix",
    "select_model",
    "train_model",
    "assign_clusters",
    "evaluate",
    "diagnostics",
    "plot_eda",
    "plot_model",
    "plot_report",
]


def _selected_stages(pipeline_config: dict, from_stage: str | None, to_stage: str | None) -> list[str]:
    stages = pipeline_config["pipeline"]["stages"]
    start = stages.index(from_stage) if from_stage else 0
    end = stages.index(to_stage) if to_stage else len(stages) - 1
    return stages[start : end + 1]


def run_pipeline(
    config_dir: Path,
    run_id: str | None = None,
    overwrite: bool = False,
    skip_existing: bool = False,
    model_name: str | None = None,
    pipeline_name: str | None = None,
    from_stage: str | None = None,
    to_stage: str | None = None,
) -> dict[str, str]:
    configs = load_project_configs(config_dir, model_name=model_name, pipeline_name=pipeline_name)
    base_config = configs["base"]
    data_config = configs["data"]
    features_config = configs["features"]
    model_config = configs["model"]
    pipeline_config = configs["pipeline"]
    adapter = create_model_adapter(model_config["model"]["name"])

    run_id = build_run_id(run_id)
    run_paths = get_run_paths(base_config, run_id)
    logger = setup_logging(run_paths.log_file, level=base_config["runtime"]["logging_level"])

    effective_overwrite = overwrite or base_config["runtime"]["overwrite"]
    effective_skip_existing = skip_existing or base_config["runtime"]["skip_existing"]

    seed = int(base_config["runtime"]["random_seed"])
    set_global_seed(seed)
    logger.info("Run started | run_id=%s | seed=%s", run_id, seed)
    logger.info("Git commit hash=%s", get_git_commit_hash())
    logger.info("Active pipeline=%s | active model=%s", pipeline_config["pipeline"]["name"], adapter.model_name)

    snapshot_configs(
        configs,
        run_paths.run_configs_dir / f"{run_id}.json",
    )

    interim_dir = resolve_path(base_config["paths"]["interim_data_dir"])
    processed_dir = resolve_path(base_config["paths"]["processed_data_dir"])
    selected_stages = _selected_stages(pipeline_config, from_stage, to_stage)

    artifacts: dict[str, str] = {}
    for stage in selected_stages:
        logger.info("Stage start: %s", stage)
        if stage in {"ingest", "profile", "validate", "clean", "transform", "build_matrix"}:
            result = data_pipeline.run_data_pipeline(
                config_dir=config_dir,
                stage=stage,
                run_id=run_id,
                overwrite=effective_overwrite,
                skip_existing=effective_skip_existing,
            )
        elif stage in {"select_model", "train_model", "assign_clusters", "evaluate", "diagnostics"}:
            result = modeling_pipeline.run_stage(
                stage,
                adapter,
                model_config,
                base_config,
                run_paths.checkpoints_dir,
                run_paths.models_dir,
                run_paths.metrics_dir,
                processed_dir,
                run_paths.reports_dir,
                logger,
                effective_overwrite,
            )
        else:
            result = reporting_pipeline.run_stage(
                stage,
                adapter,
                interim_dir,
                processed_dir,
                run_paths.checkpoints_dir,
                run_paths.metrics_dir,
                run_paths.reports_dir,
                run_paths.plots_dir,
                logger,
                effective_overwrite,
            )
        artifacts.update({f"{stage}_{key}": value for key, value in result.items()})
        logger.info("Stage end: %s", stage)

    if effective_skip_existing:
        logger.info("skip_existing requested; stage modules will preserve existing outputs when possible")
    logger.info("Run completed | run_id=%s", run_id)
    return artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the anime clustering pipeline")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--pipeline", default=None, help="Pipeline config name in configs/pipelines/")
    parser.add_argument("--model", default=None, help="Model config name in configs/models/")
    parser.add_argument("--from-stage", choices=AVAILABLE_STAGES, default=None)
    parser.add_argument("--to-stage", choices=AVAILABLE_STAGES, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_pipeline(
        config_dir=resolve_path(args.config_dir),
        run_id=args.run_id,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
        model_name=args.model,
        pipeline_name=args.pipeline,
        from_stage=args.from_stage,
        to_stage=args.to_stage,
    )


if __name__ == "__main__":
    main()
