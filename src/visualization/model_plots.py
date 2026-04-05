from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt

from src.utils.io import read_dataframe


def run(
    selection_results_path: Path,
    cluster_sizes_path: Path,
    plots_dir: Path,
    logger: logging.Logger,
    overwrite: bool = False,
) -> dict[str, str]:
    bic_path = plots_dir / "gmm_bic_vs_k.png"
    aic_path = plots_dir / "gmm_aic_vs_k.png"
    loglik_path = plots_dir / "gmm_loglik_vs_k.png"
    cluster_sizes_plot = plots_dir / "gmm_cluster_sizes.png"
    if all(path.exists() for path in [bic_path, aic_path, loglik_path, cluster_sizes_plot]) and not overwrite:
        logger.info("Skipping model plots; outputs already exist")
        return {
            "bic_plot": str(bic_path),
            "aic_plot": str(aic_path),
            "loglik_plot": str(loglik_path),
            "cluster_sizes_plot": str(cluster_sizes_plot),
        }

    selection = read_dataframe(selection_results_path)
    cluster_sizes = read_dataframe(cluster_sizes_path)

    for metric, output_path, title in [
        ("bic", bic_path, "GMM BIC vs K"),
        ("aic", aic_path, "GMM AIC vs K"),
        ("log_likelihood", loglik_path, "GMM Log-Likelihood vs K"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for covariance_type, part in selection.groupby("covariance_type"):
            part = part.sort_values("k")
            ax.plot(part["k"], part[metric], marker="o", label=covariance_type)
        ax.set_title(title)
        ax.set_xlabel("K")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(cluster_sizes["cluster"].astype(str), cluster_sizes["count"])
    ax.set_title("GMM Cluster Sizes")
    fig.tight_layout()
    fig.savefig(cluster_sizes_plot, dpi=150)
    plt.close(fig)

    logger.info("Saved model plots to %s", plots_dir)
    return {
        "bic_plot": str(bic_path),
        "aic_plot": str(aic_path),
        "loglik_plot": str(loglik_path),
        "cluster_sizes_plot": str(cluster_sizes_plot),
    }
