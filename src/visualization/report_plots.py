from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from src.utils.io import read_dataframe


def run(assignments_path: Path, matrix_path: Path, plots_dir: Path, logger: logging.Logger, overwrite: bool = False) -> dict[str, str]:
    stem = assignments_path.stem.replace("_cluster_assignments", "")
    output_path = plots_dir / f"{stem}_cluster_projection.png"
    if output_path.exists() and not overwrite:
        logger.info("Skipping report plots; output already exists at %s", output_path)
        return {"plot": str(output_path)}

    assignments = read_dataframe(assignments_path)
    x = np.load(matrix_path)
    projection = PCA(n_components=2, random_state=42).fit_transform(x)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(projection[:, 0], projection[:, 1], c=assignments["cluster"], s=12, alpha=0.7, cmap="tab20")
    ax.set_title("GMM Cluster Projection")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved report plot to %s", output_path)
    return {"plot": str(output_path)}
