from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np

from src.clustering.base import ClusteringModelAdapter
from src.utils.io import write_json


def run(
    matrix_path: Path,
    adapter: ClusteringModelAdapter,
    model_config: dict,
    base_config: dict,
    models_dir: Path,
    metrics_dir: Path,
    reports_dir: Path,
    checkpoints_dir: Path,
    logger: logging.Logger,
    overwrite: bool = False,
) -> dict[str, str]:
    model_path = models_dir / f"final_{adapter.model_name}.joblib"
    metadata_path = models_dir / f"{adapter.model_name}_model_metadata.json"
    summary_path = reports_dir / f"final_{adapter.model_name}_summary.json"
    feature_names_path = checkpoints_dir / "feature_names.json"
    best_path = metrics_dir / f"{adapter.model_name}_best_params.json"
    if model_path.exists() and metadata_path.exists() and summary_path.exists() and not overwrite:
        logger.info("Skipping model training; outputs already exist")
        return {"model": str(model_path), "metadata": str(metadata_path), "summary": str(summary_path)}

    x = np.load(matrix_path)
    if not np.isfinite(x).all():
        raise ValueError("Training matrix contains NaN or infinite values")

    seed = int(base_config["runtime"]["random_seed"])
    training_params = dict(model_config["training"])
    final_override = model_config.get("final_model", {})
    if best_path.exists():
        training_params.update(json.loads(best_path.read_text(encoding="utf-8")))
    else:
        training_params.update(adapter.selection_candidates(model_config)[0])
    if final_override.get("k") is not None:
        training_params["k"] = final_override["k"]
    if final_override.get("covariance_type") is not None:
        training_params["covariance_type"] = final_override["covariance_type"]

    model = adapter.fit_final(x, training_params, seed)
    diagnostics = adapter.diagnostics(model)
    diagnostics["model_name"] = adapter.model_name
    diagnostics["training_params"] = training_params
    diagnostics["random_seed"] = seed
    diagnostics["sample_count"] = int(x.shape[0])
    diagnostics["feature_count"] = int(x.shape[1])
    diagnostics["run_id"] = base_config["experiment"].get("active_pipeline")
    if feature_names_path.exists():
        diagnostics["feature_names_path"] = str(feature_names_path)

    summary = {
        "chosen_k": int(training_params["k"]),
        "covariance_type": training_params["covariance_type"],
        "random_seed": seed,
        "n_init": int(training_params["n_init"]),
        "max_iter": int(training_params["max_iter"]),
        "tol": float(training_params["tol"]),
        "reg_covar": float(training_params["reg_covar"]),
        "converged": diagnostics["converged"],
        "n_iter": diagnostics["n_iter"],
        "lower_bound": diagnostics["lower_bound"],
        "feature_names_path": str(feature_names_path),
    }

    joblib.dump(model, model_path)
    write_json(diagnostics, metadata_path)
    write_json(summary, summary_path)
    logger.info("Saved trained %s model to %s", adapter.model_name, model_path)
    return {"model": str(model_path), "metadata": str(metadata_path), "summary": str(summary_path)}
