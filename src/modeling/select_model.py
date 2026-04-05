from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.clustering.base import ClusteringModelAdapter
from src.utils.io import write_dataframe, write_json


def _selection_guardrails(model_config: dict) -> dict:
    return {
        "reject_smallest_cluster_below": float(
            model_config.get("selection", {}).get("guardrails", {}).get("reject_smallest_cluster_below", 0.005)
        ),
        "warn_smallest_cluster_below": float(
            model_config.get("selection", {}).get("guardrails", {}).get("warn_smallest_cluster_below", 0.01)
        ),
        "tiny_cluster_threshold": float(
            model_config.get("selection", {}).get("guardrails", {}).get("tiny_cluster_threshold", 0.01)
        ),
        "reject_if_degenerate": bool(
            model_config.get("selection", {}).get("guardrails", {}).get("reject_if_degenerate", True)
        ),
        "stability_warn_below": float(
            model_config.get("selection", {}).get("guardrails", {}).get("stability_warn_below", 0.6)
        ),
    }


def _stability_config(model_config: dict) -> dict:
    return {
        "enabled": bool(model_config.get("selection", {}).get("stability", {}).get("enabled", True)),
        "n_seeds": int(model_config.get("selection", {}).get("stability", {}).get("n_seeds", 5)),
        "seed_stride": int(model_config.get("selection", {}).get("stability", {}).get("seed_stride", 1)),
    }


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


def _seed_sequence(base_seed: int, stability_cfg: dict) -> list[int]:
    if not stability_cfg["enabled"]:
        return [base_seed]
    return [base_seed + (index * stability_cfg["seed_stride"]) for index in range(stability_cfg["n_seeds"])]


def _evaluate_candidate_stability(
    adapter: ClusteringModelAdapter,
    x: np.ndarray,
    candidate: dict,
    seeds: list[int],
) -> tuple[dict, list[np.ndarray]]:
    run_metrics: list[dict] = []
    labels_per_seed: list[np.ndarray] = []
    for seed in seeds:
        model, metrics = adapter.fit_candidate(x, candidate, seed)
        labels, _ = adapter.predict(model, x)
        run_metrics.append(metrics)
        labels_per_seed.append(labels)

    aggregated: dict[str, object] = {}
    numeric_keys = {
        key
        for metrics in run_metrics
        for key, value in metrics.items()
        if isinstance(value, (int, float, np.integer, np.floating, bool)) or value is None
    }
    for key in sorted(numeric_keys):
        values = [metrics[key] for metrics in run_metrics if metrics.get(key) is not None]
        if not values:
            aggregated[key] = None
            continue
        if all(isinstance(value, (bool, np.bool_)) for value in values):
            aggregated[key] = bool(all(values))
            continue
        numeric_values = [float(value) for value in values]
        aggregated[key] = float(np.mean(numeric_values))
        if len(numeric_values) > 1:
            aggregated[f"{key}_std"] = float(np.std(numeric_values))

    aggregated["seed_values"] = seeds
    aggregated["seed_count"] = len(seeds)
    aggregated["random_seed"] = int(seeds[0])

    pairwise_ari: list[float] = []
    pairwise_nmi: list[float] = []
    for i in range(len(labels_per_seed)):
        for j in range(i + 1, len(labels_per_seed)):
            pairwise_ari.append(float(adjusted_rand_score(labels_per_seed[i], labels_per_seed[j])))
            pairwise_nmi.append(float(normalized_mutual_info_score(labels_per_seed[i], labels_per_seed[j])))
    aggregated["stability_ari_mean"] = float(np.mean(pairwise_ari)) if pairwise_ari else None
    aggregated["stability_ari_std"] = float(np.std(pairwise_ari)) if pairwise_ari else None
    aggregated["stability_nmi_mean"] = float(np.mean(pairwise_nmi)) if pairwise_nmi else None
    aggregated["stability_nmi_std"] = float(np.std(pairwise_nmi)) if pairwise_nmi else None
    return aggregated, labels_per_seed


def _annotate_candidate_flags(results: pd.DataFrame, guardrails: dict) -> pd.DataFrame:
    annotated = results.copy()
    annotated["passes_size_guardrail"] = (
        annotated["smallest_cluster_proportion"] >= guardrails["reject_smallest_cluster_below"]
    )
    annotated["passes_degenerate_guardrail"] = (~annotated["degenerate_component"]) | (
        not guardrails["reject_if_degenerate"]
    )
    stability = annotated["stability_ari_mean"] if "stability_ari_mean" in annotated.columns else pd.Series(1.0, index=annotated.index)
    annotated["passes_stability_guardrail"] = stability.fillna(1.0) >= guardrails["stability_warn_below"]
    annotated["selection_valid"] = annotated["passes_size_guardrail"] & annotated["passes_degenerate_guardrail"]
    return annotated


def _sort_results(results: pd.DataFrame, model_config: dict) -> pd.DataFrame:
    sort_columns, ascending = _sort_spec(model_config)
    ordered = results.copy()
    ordered["_selection_valid_rank"] = (~ordered["selection_valid"]).astype(int)
    ordered["_stability_rank"] = (~ordered["passes_stability_guardrail"]).astype(int)
    if "stability_ari_mean" in ordered.columns:
        ordered["_stability_metric"] = ordered["stability_ari_mean"].fillna(-1.0)
    else:
        ordered["_stability_metric"] = 1.0
    ordered = ordered.sort_values(
        ["_selection_valid_rank", "_stability_rank", *sort_columns, "_stability_metric"],
        ascending=[True, True, *ascending, False],
        na_position="last",
    ).reset_index(drop=True)
    return ordered.drop(columns=["_selection_valid_rank", "_stability_rank", "_stability_metric"])


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
    if best_row.get("stability_ari_mean") is not None:
        lines.append(f"- Mean stability ARI across seeds: {best_row['stability_ari_mean']:.4f}.")
    if best_row.get("stability_nmi_mean") is not None:
        lines.append(f"- Mean stability NMI across seeds: {best_row['stability_nmi_mean']:.4f}.")
    lines.append("- Selection rationale: rank by the configured metrics, then prefer stable, non-degenerate, non-tiny-cluster solutions.")
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
    guardrails = _selection_guardrails(model_config)
    stability_cfg = _stability_config(model_config)
    seeds = _seed_sequence(seed, stability_cfg)
    rows = []
    warnings: list[str] = []
    for candidate in adapter.selection_candidates(model_config):
        metrics, _ = _evaluate_candidate_stability(adapter, x, candidate, seeds)
        if metrics.get("degenerate_component"):
            context = f" cov={metrics['covariance_type']}" if metrics.get("covariance_type") is not None else ""
            warnings.append(f"Potentially degenerate component for k={metrics['k']}{context}")
        if metrics.get("smallest_cluster_proportion", 1.0) < guardrails["warn_smallest_cluster_below"]:
            context = f" cov={metrics['covariance_type']}" if metrics.get("covariance_type") is not None else ""
            warnings.append(f"Extreme imbalance for k={metrics['k']}{context}: smallest cluster proportion {metrics['smallest_cluster_proportion']:.4f}")
        if metrics.get("stability_ari_mean") is not None and metrics["stability_ari_mean"] < guardrails["stability_warn_below"]:
            context = f" cov={metrics['covariance_type']}" if metrics.get("covariance_type") is not None else ""
            warnings.append(f"Low stability for k={metrics['k']}{context}: mean ARI {metrics['stability_ari_mean']:.4f}")
        rows.append(metrics)

    results = _annotate_candidate_flags(pd.DataFrame(rows), guardrails)
    results = _sort_results(results, model_config)
    best_params = adapter.pick_best_params(results, model_config)
    best_row = adapter.best_row(results, best_params)
    summary = {
        "selection_metric_primary": model_config["selection"]["scoring"]["primary"],
        "selection_metric_secondary": model_config["selection"]["scoring"]["secondary"],
        "selection_metric_tertiary": model_config["selection"]["scoring"].get("tertiary"),
        "guardrails": guardrails,
        "stability": stability_cfg,
        "best_params": best_params,
        "best_metrics": best_row,
        "candidate_count": int(len(results)),
        "valid_candidate_count": int(results["selection_valid"].sum()) if "selection_valid" in results.columns else int(len(results)),
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
