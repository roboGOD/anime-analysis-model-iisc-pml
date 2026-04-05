from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt

from src.utils.io import read_dataframe


def run(
    model_name: str,
    selection_results_path: Path,
    cluster_sizes_path: Path,
    plots_dir: Path,
    logger: logging.Logger,
    overwrite: bool = False,
) -> dict[str, str]:
    cluster_sizes_plot = plots_dir / f"{model_name}_cluster_sizes.png"
    selection = read_dataframe(selection_results_path)
    cluster_sizes = read_dataframe(cluster_sizes_path)
    group_column = "covariance_type" if "covariance_type" in selection.columns else None
    metric_specs = []
    if {"bic", "aic", "log_likelihood"}.issubset(selection.columns):
        metric_specs = [
            ("bic", plots_dir / f"{model_name}_bic_vs_k.png", f"{model_name.upper()} BIC vs K", "bic_plot"),
            ("aic", plots_dir / f"{model_name}_aic_vs_k.png", f"{model_name.upper()} AIC vs K", "aic_plot"),
            ("log_likelihood", plots_dir / f"{model_name}_loglik_vs_k.png", f"{model_name.upper()} Log-Likelihood vs K", "loglik_plot"),
        ]
        group_column = "covariance_type"
    else:
        metric_map = {
            "silhouette": "Silhouette vs K",
            "davies_bouldin": "Davies-Bouldin vs K",
            "calinski_harabasz": "Calinski-Harabasz vs K",
            "inertia": "Inertia vs K",
        }
        metric_specs = [
            (metric, plots_dir / f"{model_name}_{metric}_vs_k.png", f"{model_name.upper()} {title}", f"{metric}_plot")
            for metric, title in metric_map.items()
            if metric in selection.columns
        ]

    required_paths = [cluster_sizes_plot] + [path for _, path, _, _ in metric_specs]
    if all(path.exists() for path in required_paths) and not overwrite:
        logger.info("Skipping model plots; outputs already exist")
        outputs = {"cluster_sizes_plot": str(cluster_sizes_plot)}
        for _, path, _, key in metric_specs:
            outputs[key] = str(path)
        return outputs

    outputs: dict[str, str] = {}
    for metric, output_path, title, key in metric_specs:
        fig, ax = plt.subplots(figsize=(8, 5))
        if group_column is not None:
            for group_value, part in selection.groupby(group_column):
                part = part.sort_values("k")
                ax.plot(part["k"], part[metric], marker="o", label=str(group_value))
            ax.legend()
        else:
            part = selection.sort_values("k")
            ax.plot(part["k"], part[metric], marker="o", color="#4e79a7")
        ax.set_title(title)
        ax.set_xlabel("K")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        outputs[key] = str(output_path)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(cluster_sizes["cluster"].astype(str), cluster_sizes["count"])
    ax.set_title(f"{model_name.upper()} Cluster Sizes")
    fig.tight_layout()
    fig.savefig(cluster_sizes_plot, dpi=150)
    plt.close(fig)

    logger.info("Saved model plots to %s", plots_dir)
    outputs["cluster_sizes_plot"] = str(cluster_sizes_plot)
    return outputs
