from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.io import read_dataframe


def _save_hist(series: pd.Series, output_path: Path, title: str, bins: int = 30, log1p: bool = False) -> None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if log1p:
        values = np.log1p(values.clip(lower=0))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=bins, color="#4472c4", edgecolor="white")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_bar(series: pd.Series, output_path: Path, title: str, top_n: int = 12) -> None:
    counts = series.fillna("missing").astype(str).value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.index.astype(str), counts.values, color="#70ad47")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_profile_plots(ingested_path: Path, plots_dir: Path, logger: logging.Logger, overwrite: bool = False) -> dict[str, str]:
    outputs = {
        "score": plots_dir / "eda_score_hist.png",
        "episodes": plots_dir / "eda_episodes_hist.png",
        "duration": plots_dir / "eda_duration_hist.png",
        "members": plots_dir / "eda_log_members_hist.png",
        "favorites": plots_dir / "eda_log_favorites_hist.png",
        "type": plots_dir / "eda_top_types.png",
        "source": plots_dir / "eda_top_sources.png",
        "rating": plots_dir / "eda_top_ratings.png",
        "genres": plots_dir / "eda_top_genres.png",
        "missing": plots_dir / "eda_missing_values.png",
    }
    if all(path.exists() for path in outputs.values()) and not overwrite:
        logger.info("Skipping profile plots; outputs already exist")
        return {key: str(value) for key, value in outputs.items()}

    df = read_dataframe(ingested_path)
    _save_hist(df["score"], outputs["score"], "Score Distribution")
    _save_hist(df["episodes"], outputs["episodes"], "Episodes Distribution")
    if "duration" in df.columns:
        duration = df["duration"].astype(str).str.extract(r"(\d+)")[0]
        _save_hist(duration, outputs["duration"], "Duration Distribution")
    _save_hist(df["members"], outputs["members"], "Log Members Distribution", log1p=True)
    if "favorites" in df.columns:
        _save_hist(df["favorites"], outputs["favorites"], "Log Favorites Distribution", log1p=True)
    _save_bar(df["type"], outputs["type"], "Top Anime Types")
    _save_bar(df["source"], outputs["source"], "Top Sources")
    _save_bar(df["rating"], outputs["rating"], "Top Ratings")
    genre_counts = (
        df["genres"]
        .fillna("")
        .astype(str)
        .str.split(",")
        .explode()
        .str.strip()
        .replace("", pd.NA)
        .dropna()
    )
    _save_bar(genre_counts, outputs["genres"], "Top Genres")

    missing_counts = df.isna().sum().sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(missing_counts.index.astype(str), missing_counts.values, color="#c0504d")
    ax.set_title("Missing Values")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(outputs["missing"], dpi=150)
    plt.close(fig)

    logger.info("Saved profile EDA plots to %s", plots_dir)
    return {key: str(value) for key, value in outputs.items()}


def run_transformed_plots(transformed_path: Path, plots_dir: Path, logger: logging.Logger, overwrite: bool = False) -> dict[str, str]:
    outputs = {
        "transformed_numeric": plots_dir / "eda_transformed_numeric.png",
        "correlation": plots_dir / "eda_numeric_correlation_heatmap.png",
        "feature_families": plots_dir / "eda_feature_family_counts.png",
    }
    if all(path.exists() for path in outputs.values()) and not overwrite:
        logger.info("Skipping transformed plots; outputs already exist")
        return {key: str(value) for key, value in outputs.items()}

    df = read_dataframe(transformed_path)
    numeric_candidates = [column for column in ["score", "episodes", "duration_minutes", "members_log1p", "favorites_log1p", "scored_by_log1p"] if column in df.columns]
    if numeric_candidates:
        fig, axes = plt.subplots(1, len(numeric_candidates), figsize=(5 * len(numeric_candidates), 4))
        axes = [axes] if len(numeric_candidates) == 1 else axes
        for ax, column in zip(axes, numeric_candidates, strict=False):
            ax.hist(pd.to_numeric(df[column], errors="coerce").dropna(), bins=30, color="#5b9bd5", edgecolor="white")
            ax.set_title(column)
        fig.tight_layout()
        fig.savefig(outputs["transformed_numeric"], dpi=150)
        plt.close(fig)

        corr = df[numeric_candidates].apply(pd.to_numeric, errors="coerce").corr().fillna(0)
        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index)
        ax.set_title("Numeric Correlation Heatmap")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(outputs["correlation"], dpi=150)
        plt.close(fig)

    counts = {
        "numeric": len([column for column in df.columns if column.endswith("_log1p") or column in {"score", "episodes", "rank", "popularity", "duration_minutes"}]),
        "one_hot categorical": len([column for column in df.columns if column in {"type", "source", "rating", "status", "premiered_season"}]),
        "multi_hot tags": len([column for column in df.columns if "__" in column]),
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(counts.keys()), list(counts.values()), color=["#4f81bd", "#9bbb59", "#f79646"])
    ax.set_title("Feature Family Counts")
    fig.tight_layout()
    fig.savefig(outputs["feature_families"], dpi=150)
    plt.close(fig)

    logger.info("Saved transformed data plots to %s", plots_dir)
    return {key: str(value) for key, value in outputs.items()}
