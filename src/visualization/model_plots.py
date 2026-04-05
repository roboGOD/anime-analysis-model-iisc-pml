from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt

from src.utils.io import read_dataframe


def run(selection_results_path: Path, cluster_sizes_path: Path, plots_dir: Path, logger: logging.Logger, overwrite: bool = False) -> dict[str, str]:
    stem = selection_results_path.stem.replace("_model_selection", "")
    output_path = plots_dir / f"{stem}_model_selection.png"
    if output_path.exists() and not overwrite:
        logger.info("Skipping model plots; output already exists at %s", output_path)
        return {"plot": str(output_path)}

    selection = read_dataframe(selection_results_path)
    cluster_sizes = read_dataframe(cluster_sizes_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for covariance_type, part in selection.groupby("covariance_type"):
        axes[0].plot(part["k"], part["bic"], marker="o", label=f"BIC {covariance_type}")
        axes[1].plot(part["k"], part["aic"], marker="o", label=f"AIC {covariance_type}")
    axes[0].legend()
    axes[1].legend()
    axes[0].set_title("BIC by K")
    axes[1].set_title("AIC by K")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    size_path = plots_dir / f"{stem}_cluster_sizes.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(cluster_sizes["cluster"].astype(str), cluster_sizes["count"])
    ax.set_title("Cluster Sizes")
    fig.tight_layout()
    fig.savefig(size_path, dpi=150)
    plt.close(fig)

    logger.info("Saved model plots to %s and %s", output_path, size_path)
    return {"selection_plot": str(output_path), "cluster_sizes_plot": str(size_path)}
