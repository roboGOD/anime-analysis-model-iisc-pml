from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils.io import read_dataframe, write_dataframe, write_json


def run(
    assignments_path: Path,
    metrics_dir: Path,
    logger: logging.Logger,
    overwrite: bool = False,
) -> dict[str, str]:
    stem = assignments_path.stem.replace("_cluster_assignments", "")
    json_path = metrics_dir / f"{stem}_cluster_metrics.json"
    csv_path = metrics_dir / f"{stem}_cluster_sizes.csv"
    if json_path.exists() and csv_path.exists() and not overwrite:
        logger.info("Skipping evaluation; outputs already exist")
        return {"metrics": str(json_path), "cluster_sizes": str(csv_path)}

    df = read_dataframe(assignments_path)
    cluster_sizes = df["cluster"].value_counts().sort_index().rename_axis("cluster").reset_index(name="count")
    metrics = {
        "n_titles": int(len(df)),
        "n_clusters": int(cluster_sizes.shape[0]),
        "avg_assignment_confidence": float(df["max_probability"].mean()),
        "min_assignment_confidence": float(df["max_probability"].min()),
        "max_assignment_confidence": float(df["max_probability"].max()),
    }
    write_json(metrics, json_path)
    write_dataframe(cluster_sizes, csv_path)
    logger.info("Saved evaluation metrics to %s", json_path)
    return {"metrics": str(json_path), "cluster_sizes": str(csv_path)}
