from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.clustering.base import ClusteringModelAdapter
from src.utils.io import read_dataframe, write_dataframe


def run(
    row_mapping_path: Path,
    matrix_path: Path,
    model_path: Path,
    adapter: ClusteringModelAdapter,
    reports_dir: Path,
    logger: logging.Logger,
    overwrite: bool = False,
) -> dict[str, str]:
    output_path = reports_dir / "cluster_assignments.parquet"
    csv_path = reports_dir / "cluster_assignments.csv"
    model_specific_path = reports_dir / f"{adapter.model_name}_cluster_assignments.parquet"
    if output_path.exists() and csv_path.exists() and model_specific_path.exists() and not overwrite:
        logger.info("Skipping cluster assignment; outputs already exist")
        return {"parquet": str(output_path), "csv": str(csv_path), "model_specific": str(model_specific_path)}

    row_mapping = read_dataframe(row_mapping_path)
    x = np.load(matrix_path)
    model = joblib.load(model_path)
    assignments, probabilities = adapter.predict(model, x)
    if probabilities is None:
        raise ValueError("GMM assignments require soft probabilities")

    entropy = -(probabilities * np.log(probabilities + 1e-12)).sum(axis=1)
    output = row_mapping.copy()
    output["cluster"] = assignments
    output["max_probability"] = probabilities.max(axis=1)
    output["assignment_entropy"] = entropy
    for index in range(probabilities.shape[1]):
        output[f"cluster_prob_{index}"] = probabilities[:, index]

    write_dataframe(output, output_path)
    write_dataframe(output, csv_path)
    write_dataframe(output, model_specific_path)
    logger.info("Saved %s cluster assignments to %s", adapter.model_name, output_path)
    return {"parquet": str(output_path), "csv": str(csv_path), "model_specific": str(model_specific_path)}
