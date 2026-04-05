from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.utils.io import read_dataframe


def run(
    assignments_path: Path,
    matrix_path: Path,
    cleaned_metadata_path: Path,
    plots_dir: Path,
    logger: logging.Logger,
    overwrite: bool = False,
) -> dict[str, str]:
    confidence_path = plots_dir / "gmm_max_responsibility_hist.png"
    entropy_path = plots_dir / "gmm_assignment_entropy_hist.png"
    heatmap_path = plots_dir / "gmm_cluster_numeric_heatmap.png"
    genres_path = plots_dir / "gmm_top_genres_by_cluster.png"
    categories_path = plots_dir / "gmm_top_categorical_by_cluster.png"
    projection_path = plots_dir / "gmm_cluster_projection.png"
    if all(path.exists() for path in [confidence_path, entropy_path, heatmap_path, genres_path, categories_path, projection_path]) and not overwrite:
        logger.info("Skipping report plots; outputs already exist")
        return {
            "confidence_plot": str(confidence_path),
            "entropy_plot": str(entropy_path),
            "heatmap_plot": str(heatmap_path),
            "genres_plot": str(genres_path),
            "categories_plot": str(categories_path),
            "projection_plot": str(projection_path),
        }

    assignments = read_dataframe(assignments_path)
    cleaned = read_dataframe(cleaned_metadata_path)
    merged = assignments.merge(cleaned, on=["anime_id", "name"], how="left")
    x = np.load(matrix_path)

    for column, output_path, title in [
        ("max_probability", confidence_path, "Max Responsibility"),
        ("assignment_entropy", entropy_path, "Assignment Entropy"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(merged[column].dropna(), bins=30, color="#5b9bd5", edgecolor="white")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    numeric_columns = [column for column in ["score", "episodes", "duration_minutes", "members", "favorites", "scored_by"] if column in merged.columns]
    if numeric_columns:
        heatmap_data = merged.groupby("cluster")[numeric_columns].mean()
        values = heatmap_data.to_numpy(dtype=float)
        row_means = values.mean(axis=0, keepdims=True)
        row_stds = values.std(axis=0, keepdims=True)
        z_scores = (values - row_means) / np.where(row_stds == 0, 1, row_stds)
        fig, ax = plt.subplots(figsize=(8, 5))
        image = ax.imshow(z_scores, cmap="coolwarm", aspect="auto")
        ax.set_xticks(range(len(numeric_columns)))
        ax.set_xticklabels(numeric_columns, rotation=45, ha="right")
        ax.set_yticks(range(len(heatmap_data.index)))
        ax.set_yticklabels([f"Cluster {idx}" for idx in heatmap_data.index])
        ax.set_title("Cluster Numeric Feature Z-Scores")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(heatmap_path, dpi=150)
        plt.close(fig)

    genres = (
        merged[["cluster", "genres"]]
        .assign(genres=lambda frame: frame["genres"].fillna("missing").astype(str).str.split("|"))
        .explode("genres")
    )
    genres = genres[genres["genres"].notna() & (genres["genres"] != "missing")]
    genre_counts = genres.groupby(["cluster", "genres"]).size().reset_index(name="count")
    genre_counts = genre_counts.sort_values(["cluster", "count"], ascending=[True, False]).groupby("cluster").head(3)
    if not genre_counts.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = [f"C{row.cluster}:{row.genres}" for row in genre_counts.itertuples()]
        ax.bar(labels, genre_counts["count"], color="#70ad47")
        ax.set_title("Top Genres by Cluster")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(genres_path, dpi=150)
        plt.close(fig)

    categorical_rows = []
    for column in [col for col in ["type", "source", "rating"] if col in merged.columns]:
        counts = merged.groupby(["cluster", column]).size().reset_index(name="count")
        top = counts.sort_values(["cluster", "count"], ascending=[True, False]).groupby("cluster").head(1)
        top["label"] = top[column].astype(str)
        categorical_rows.append(top[["cluster", "label", "count"]])
    if categorical_rows:
        top_categories = pd.concat(categorical_rows, ignore_index=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = [f"C{row.cluster}:{row.label}" for row in top_categories.itertuples()]
        ax.bar(labels, top_categories["count"], color="#f79646")
        ax.set_title("Top Categorical Levels by Cluster")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(categories_path, dpi=150)
        plt.close(fig)

    projection = PCA(n_components=2, random_state=42).fit_transform(x)
    alpha = np.clip(merged["max_probability"].to_numpy(), 0.2, 1.0)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(projection[:, 0], projection[:, 1], c=merged["cluster"], s=14, alpha=alpha, cmap="tab20")
    ax.set_title("GMM Cluster Projection")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(projection_path, dpi=150)
    plt.close(fig)

    logger.info("Saved report plots to %s", plots_dir)
    return {
        "confidence_plot": str(confidence_path),
        "entropy_plot": str(entropy_path),
        "heatmap_plot": str(heatmap_path),
        "genres_plot": str(genres_path),
        "categories_plot": str(categories_path),
        "projection_plot": str(projection_path),
    }
