from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.utils.io import read_dataframe, write_json


def run(
    model_metadata_path: Path,
    assignments_path: Path,
    reports_dir: Path,
    logger: logging.Logger,
    overwrite: bool = False,
) -> dict[str, str]:
    model_name = model_metadata_path.stem.removesuffix("_model_metadata")
    output_path = reports_dir / f"{model_name}_diagnostics.json"
    if output_path.exists() and not overwrite:
        logger.info("Skipping diagnostics; output already exists at %s", output_path)
        return {"diagnostics": str(output_path)}

    metadata = json.loads(model_metadata_path.read_text(encoding="utf-8"))
    assignments = read_dataframe(assignments_path)
    prob_cols = [column for column in assignments.columns if column.startswith("cluster_prob_")]
    low_confidence_fraction = float((assignments["max_probability"] < 0.5).mean()) if "max_probability" in assignments.columns else None
    weights = np.asarray(metadata.get("weights", []), dtype=float)
    means = np.asarray(metadata.get("means", []), dtype=float)
    duplicate_cluster_warning = False
    if means.ndim == 2 and len(means) > 1:
        distances = []
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                distances.append(float(np.linalg.norm(means[i] - means[j])))
        duplicate_cluster_warning = bool(distances and min(distances) < 0.5)

    diagnostics = {
        "model_name": metadata["model_name"],
        "converged": metadata.get("converged"),
        "n_iter": metadata.get("n_iter"),
        "lower_bound": metadata.get("lower_bound"),
        "inertia": metadata.get("inertia"),
        "component_weight_distribution": weights.tolist(),
        "smallest_component_weight": float(weights.min()) if weights.size else None,
        "near_zero_weight_component": bool(weights.size and np.any(weights < 0.01)),
        "covariance_valid": bool(np.all(np.asarray(metadata.get("covariances", []), dtype=float) >= 0)) if metadata.get("covariances") else True,
        "mean_assignment_entropy": float(assignments["assignment_entropy"].mean()) if "assignment_entropy" in assignments.columns else None,
        "low_confidence_fraction": low_confidence_fraction,
        "duplicate_cluster_warning": duplicate_cluster_warning,
        "responsibility_columns": prob_cols,
        "sample_size_vs_model_complexity_warning": bool(
            metadata.get("sample_count", 0) < max(10 * metadata.get("feature_count", 1), 5 * metadata.get("component_count", 1))
        ),
    }
    write_json(diagnostics, output_path)
    logger.info("Saved diagnostics to %s", output_path)
    return {"diagnostics": str(output_path)}
