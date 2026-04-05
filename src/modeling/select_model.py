from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.clustering.base import ClusteringModelAdapter
from src.utils.io import write_dataframe, write_json


def _sort_spec(model_config: dict) -> tuple[list[str], list[bool]]:
    scoring = model_config["selection"]["scoring"]
    columns = [scoring["primary"], scoring["secondary"]]
    directions = [
        scoring.get("primary_mode", "min") == "min",
        scoring.get("secondary_mode", "min") == "min",
    ]
    if scoring.get("tertiary") is not None:
        columns.append(scoring["tertiary"])
        directions.append(scoring.get("tertiary_mode", "min") == "min")
    return columns, directions


def _describe_best_choice(adapter: ClusteringModelAdapter, best_params: dict, best_row: dict, scoring: dict) -> list[str]:
    selected_bits = [f"`K={best_params['k']}`"]
    if "covariance_type" in best_params:
        selected_bits.append(f"covariance `{best_params['covariance_type']}`")

    lines = [
        f"# {adapter.model_name.upper()} Model Selection Summary",
        f"- Selected {' with '.join(selected_bits)}.",
    ]
    for label, mode_key in [
        (scoring["primary"], "primary_mode"),
        (scoring["secondary"], "secondary_mode"),
        (scoring.get("tertiary"), "tertiary_mode"),
    ]:
        if label is None or label not in best_row or best_row[label] is None:
            continue
        direction = "minimize" if scoring.get(mode_key, "min") == "min" else "maximize"
        lines.append(f"- {label}: {best_row[label]:.6f} ({direction}).")
    if "smallest_cluster_proportion" in best_row:
        lines.append(f"- Smallest cluster proportion: {best_row['smallest_cluster_proportion']:.4f}.")
    lines.append("- Selection rationale: apply the configured ranking metrics, then reject obviously imbalanced or degenerate solutions.")
    return lines


def run(
    matrix_path: Path,
    adapter: ClusteringModelAdapter,
    model_config: dict,
    base_config: dict,
    metrics_dir: Path,
    reports_dir: Path,
    logger: logging.Logger,
    overwrite: bool = False,
) -> dict[str, str]:
    results_path = metrics_dir / f"{adapter.model_name}_model_selection.csv"
    best_path = metrics_dir / f"{adapter.model_name}_best_params.json"
    summary_path = metrics_dir / f"{adapter.model_name}_model_selection.json"
    rationale_path = reports_dir / f"{adapter.model_name}_selection_summary.md"
    if results_path.exists() and best_path.exists() and summary_path.exists() and not overwrite:
        logger.info("Skipping model selection; outputs already exist")
        return {
            "results": str(results_path),
            "best": str(best_path),
            "summary": str(summary_path),
            "rationale": str(rationale_path),
        }

    x = np.load(matrix_path)
    seed = int(base_config["runtime"]["random_seed"])
    rows = []
    warnings: list[str] = []
    for candidate in adapter.selection_candidates(model_config):
        _, metrics = adapter.fit_candidate(x, candidate, seed)
        if metrics.get("degenerate_component"):
            context = f" cov={metrics['covariance_type']}" if metrics.get("covariance_type") is not None else ""
            warnings.append(f"Potentially degenerate component for k={metrics['k']}{context}")
        if metrics.get("smallest_cluster_proportion", 1.0) < 0.02:
            context = f" cov={metrics['covariance_type']}" if metrics.get("covariance_type") is not None else ""
            warnings.append(f"Extreme imbalance for k={metrics['k']}{context}: smallest cluster proportion {metrics['smallest_cluster_proportion']:.4f}")
        rows.append(metrics)

    sort_columns, ascending = _sort_spec(model_config)
    results = pd.DataFrame(rows).sort_values(sort_columns, ascending=ascending, na_position="last").reset_index(drop=True)
    best_params = adapter.pick_best_params(results, model_config)
    best_row = adapter.best_row(results, best_params)
    summary = {
        "selection_metric_primary": model_config["selection"]["scoring"]["primary"],
        "selection_metric_secondary": model_config["selection"]["scoring"]["secondary"],
        "selection_metric_tertiary": model_config["selection"]["scoring"].get("tertiary"),
        "best_params": best_params,
        "best_metrics": best_row,
        "candidate_count": int(len(results)),
        "warnings": warnings,
    }
    rationale = "\n".join(_describe_best_choice(adapter, best_params, best_row, model_config["selection"]["scoring"]))

    write_dataframe(results, results_path)
    write_json(best_params, best_path)
    write_json(summary, summary_path)
    rationale_path.write_text(rationale, encoding="utf-8")
    logger.info("Saved %s model selection results to %s", adapter.model_name, results_path)
    return {
        "results": str(results_path),
        "best": str(best_path),
        "summary": str(summary_path),
        "rationale": str(rationale_path),
    }
