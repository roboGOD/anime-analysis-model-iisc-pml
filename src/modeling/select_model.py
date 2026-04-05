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
            warnings.append(
                f"Potentially degenerate component for k={metrics['k']} cov={metrics['covariance_type']}"
            )
        if metrics.get("smallest_cluster_proportion", 1.0) < 0.02:
            warnings.append(
                f"Extreme imbalance for k={metrics['k']} cov={metrics['covariance_type']}: smallest cluster proportion {metrics['smallest_cluster_proportion']:.4f}"
            )
        rows.append(metrics)

    results = pd.DataFrame(rows).sort_values(["bic", "aic"], ascending=True).reset_index(drop=True)
    best_params = adapter.pick_best_params(results, model_config)
    best_row = results.iloc[0].to_dict()
    summary = {
        "selection_metric_primary": model_config["selection"]["scoring"]["primary"],
        "selection_metric_secondary": model_config["selection"]["scoring"]["secondary"],
        "best_params": best_params,
        "best_metrics": best_row,
        "candidate_count": int(len(results)),
        "warnings": warnings,
    }
    rationale = "\n".join(
        [
            f"# {adapter.model_name.upper()} Model Selection Summary",
            f"- Selected `K={best_params['k']}` with covariance `{best_params['covariance_type']}`.",
            f"- Primary criterion: BIC ({best_row['bic']:.3f}).",
            f"- Secondary criterion: AIC ({best_row['aic']:.3f}).",
            f"- Log-likelihood support: {best_row['log_likelihood']:.6f}.",
            f"- Smallest cluster proportion: {best_row['smallest_cluster_proportion']:.4f}.",
            "- Selection rationale: choose the best BIC, then check AIC and imbalance warnings to avoid obviously degenerate solutions.",
        ]
    )

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
