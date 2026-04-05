from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOGS_DIR = PROJECT_ROOT / "logs"


def build_run_id(value: str | None = None) -> str:
    return value or datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    logs_dir: Path
    checkpoints_dir: Path
    models_dir: Path
    metrics_dir: Path
    plots_dir: Path
    reports_dir: Path
    run_configs_dir: Path

    @property
    def log_file(self) -> Path:
        return self.logs_dir / f"{self.run_id}.log"


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_run_paths(base_config: dict, run_id: str) -> RunPaths:
    paths_cfg = base_config["paths"]
    run_paths = RunPaths(
        run_id=run_id,
        logs_dir=ensure_directory(resolve_path(paths_cfg["logs_dir"])),
        checkpoints_dir=ensure_directory(resolve_path(paths_cfg["checkpoints_dir"])),
        models_dir=ensure_directory(resolve_path(paths_cfg["models_dir"])),
        metrics_dir=ensure_directory(resolve_path(paths_cfg["metrics_dir"])),
        plots_dir=ensure_directory(resolve_path(paths_cfg["plots_dir"])),
        reports_dir=ensure_directory(resolve_path(paths_cfg["reports_dir"])),
        run_configs_dir=ensure_directory(resolve_path(paths_cfg["run_configs_dir"])),
    )
    ensure_directory(resolve_path(paths_cfg["raw_data_dir"]))
    ensure_directory(resolve_path(paths_cfg["interim_data_dir"]))
    ensure_directory(resolve_path(paths_cfg["processed_data_dir"]))
    return run_paths
