from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.clustering.base import ClusteringModelAdapter
from src.modeling.artifacts import model_assignments_csv_path, model_assignments_parquet_path
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
    output_path = model_assignments_parquet_path(reports_dir, adapter.model_name)
    csv_path = model_assignments_csv_path(reports_dir, adapter.model_name)
    default_output_path = reports_dir / "cluster_assignments.parquet"
    default_csv_path = reports_dir / "cluster_assignments.csv"
    required_outputs = [output_path, csv_path]
    if adapter.model_name == "gmm":
        required_outputs.extend([default_output_path, default_csv_path])
    if all(path.exists() for path in required_outputs) and not overwrite:
        logger.info("Skipping cluster assignment; outputs already exist")
        return {"parquet": str(output_path), "csv": str(csv_path)}

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
    if adapter.model_name == "gmm":
        write_dataframe(output, default_output_path)
        write_dataframe(output, default_csv_path)
    logger.info("Saved %s cluster assignments to %s", adapter.model_name, output_path)
    return {"parquet": str(output_path), "csv": str(csv_path)}
