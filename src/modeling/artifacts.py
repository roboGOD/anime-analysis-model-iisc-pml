from __future__ import annotations

from pathlib import Path


def matrix_stem(model_config: dict, model_name: str) -> str:
    return model_config.get("artifacts", {}).get("matrix_stem", f"X_{model_name}")


def matrix_npy_path(processed_dir: Path, model_config: dict, model_name: str) -> Path:
    return processed_dir / f"{matrix_stem(model_config, model_name)}.npy"


def matrix_table_path(processed_dir: Path, model_config: dict, model_name: str) -> Path:
    return processed_dir / f"{matrix_stem(model_config, model_name)}.parquet"


def full_matrix_npy_path(processed_dir: Path, model_config: dict) -> Path:
    stem = model_config.get("reduction", {}).get("source_matrix_stem", "X_gmm")
    return processed_dir / f"{stem}.npy"


def model_assignments_parquet_path(reports_dir: Path, model_name: str) -> Path:
    return reports_dir / f"{model_name}_cluster_assignments.parquet"


def model_assignments_csv_path(reports_dir: Path, model_name: str) -> Path:
    return reports_dir / f"{model_name}_cluster_assignments.csv"


def cluster_profile_csv_path(reports_dir: Path, model_name: str) -> Path:
    return reports_dir / f"{model_name}_cluster_profile_tables.csv"


def cluster_profile_md_path(reports_dir: Path, model_name: str) -> Path:
    return reports_dir / f"{model_name}_cluster_profile_summary.md"
