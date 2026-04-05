from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np

from src.clustering.base import ClusteringModelAdapter
from src.utils.io import read_dataframe, write_dataframe


def run(
    feature_table_path: Path,
    matrix_path: Path,
    model_path: Path,
    adapter: ClusteringModelAdapter,
    processed_dir: Path,
    logger: logging.Logger,
    overwrite: bool = False,
) -> dict[str, str]:
    output_path = processed_dir / f"{adapter.model_name}_cluster_assignments.parquet"
    csv_path = processed_dir / f"{adapter.model_name}_cluster_assignments.csv"
    if output_path.exists() and csv_path.exists() and not overwrite:
        logger.info("Skipping cluster assignment; outputs already exist")
        return {"parquet": str(output_path), "csv": str(csv_path)}

    feature_table = read_dataframe(feature_table_path)
    x = np.load(matrix_path)
    model = joblib.load(model_path)
    assignments, probabilities = adapter.predict(model, x)

    output = feature_table[["anime_id", "name"]].copy()
    output["cluster"] = assignments
    if probabilities is not None:
        output["max_probability"] = probabilities.max(axis=1)
        for index in range(probabilities.shape[1]):
            output[f"cluster_prob_{index}"] = probabilities[:, index]

    write_dataframe(output, output_path)
    write_dataframe(output, csv_path)
    logger.info("Saved %s cluster assignments to %s", adapter.model_name, output_path)
    return {"parquet": str(output_path), "csv": str(csv_path)}
