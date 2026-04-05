from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt

from src.utils.io import read_dataframe


def run(cleaned_path: Path, plots_dir: Path, logger: logging.Logger, overwrite: bool = False) -> dict[str, str]:
    output_path = plots_dir / "eda_overview.png"
    if output_path.exists() and not overwrite:
        logger.info("Skipping EDA plots; output already exists at %s", output_path)
        return {"plot": str(output_path)}

    df = read_dataframe(cleaned_path)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    df["score"].dropna().plot(kind="hist", bins=30, ax=axes[0], title="Score")
    df["episodes"].dropna().clip(upper=100).plot(kind="hist", bins=30, ax=axes[1], title="Episodes")
    df["type"].fillna("unknown").value_counts().head(10).plot(kind="bar", ax=axes[2], title="Type")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved EDA plot to %s", output_path)
    return {"plot": str(output_path)}
