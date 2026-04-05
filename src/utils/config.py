from __future__ import annotations

from pathlib import Path

import yaml

from src.utils.io import write_json


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_project_configs(config_dir: Path, model_name: str | None = None, pipeline_name: str | None = None) -> dict:
    base = load_yaml(config_dir / "base.yaml")
    model_name = model_name or base["experiment"]["active_model"]
    pipeline_name = pipeline_name or base["experiment"]["active_pipeline"]
    base["experiment"]["active_model"] = model_name
    base["experiment"]["active_pipeline"] = pipeline_name
    return {
        "base": base,
        "data": load_yaml(config_dir / "data.yaml"),
        "features": load_yaml(config_dir / "features.yaml"),
        "model": load_yaml(config_dir / "models" / f"{model_name}.yaml"),
        "pipeline": load_yaml(config_dir / "pipelines" / f"{pipeline_name}.yaml"),
    }


def snapshot_configs(configs: dict, output_path: Path) -> None:
    write_json(configs, output_path, indent=2)
