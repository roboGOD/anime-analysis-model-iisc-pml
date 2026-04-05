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
    logger: logging.Logger,
    overwrite: bool = False,
) -> dict[str, str]:
    model_path = models_dir / f"{adapter.model_name}_model.joblib"
    metadata_path = models_dir / f"{adapter.model_name}_model_metadata.json"
    best_path = metrics_dir / f"{adapter.model_name}_best_params.json"
    if model_path.exists() and metadata_path.exists() and not overwrite:
        logger.info("Skipping model training; outputs already exist")
        return {"model": str(model_path), "metadata": str(metadata_path)}

    x = np.load(matrix_path)
    seed = int(base_config["runtime"]["random_seed"])
    training_params = dict(model_config["training"])
    if best_path.exists():
        training_params.update(json.loads(best_path.read_text(encoding="utf-8")))
    else:
        training_params.update(adapter.selection_candidates(model_config)[0])

    model = adapter.fit_final(x, training_params, seed)
    diagnostics = adapter.diagnostics(model)
    diagnostics["model_name"] = adapter.model_name
    diagnostics["training_params"] = training_params

    joblib.dump(model, model_path)
    write_json(diagnostics, metadata_path)
    logger.info("Saved trained %s model to %s", adapter.model_name, model_path)
    return {"model": str(model_path), "metadata": str(metadata_path)}
