from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.clustering.base import ClusteringModelAdapter
from src.utils.io import write_dataframe, write_json


def run(
    matrix_path: Path,
    adapter: ClusteringModelAdapter,
    model_config: dict,
    base_config: dict,
    metrics_dir: Path,
    logger: logging.Logger,
    overwrite: bool = False,
) -> dict[str, str]:
    results_path = metrics_dir / f"{adapter.model_name}_model_selection.csv"
    best_path = metrics_dir / f"{adapter.model_name}_best_params.json"
    if results_path.exists() and best_path.exists() and not overwrite:
        logger.info("Skipping model selection; outputs already exist")
        return {"results": str(results_path), "best": str(best_path)}

    x = np.load(matrix_path)
    seed = int(base_config["runtime"]["random_seed"])
    rows = []
    for candidate in adapter.selection_candidates(model_config):
        _, metrics = adapter.fit_candidate(x, candidate, seed)
        rows.append(metrics)

    results = pd.DataFrame(rows)
    best_params = adapter.pick_best_params(results, model_config)
    write_dataframe(results, results_path)
    write_json(best_params, best_path)
    logger.info("Saved %s model selection results to %s", adapter.model_name, results_path)
    return {"results": str(results_path), "best": str(best_path)}
