from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from src.utils.io import read_dataframe, write_dataframe, write_json


def _top_probability_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column.startswith("cluster_prob_")]


def run(
    assignments_path: Path,
    matrix_path: Path,
    selection_summary_path: Path,
    cleaned_metadata_path: Path,
    metrics_dir: Path,
    reports_dir: Path,
    logger: logging.Logger,
    overwrite: bool = False,
) -> dict[str, str]:
    metrics_path = metrics_dir / "final_gmm_metrics.json"
    profile_csv_path = reports_dir / "cluster_profile_tables.csv"
    profile_md_path = reports_dir / "cluster_profile_summary.md"
    cluster_sizes_path = metrics_dir / "gmm_cluster_sizes.csv"
    if metrics_path.exists() and profile_csv_path.exists() and profile_md_path.exists() and not overwrite:
        logger.info("Skipping evaluation; outputs already exist")
        return {
            "metrics": str(metrics_path),
            "cluster_sizes": str(cluster_sizes_path),
            "profile_csv": str(profile_csv_path),
            "profile_md": str(profile_md_path),
        }

    assignments = read_dataframe(assignments_path)
    cleaned = read_dataframe(cleaned_metadata_path)
    x = np.load(matrix_path)
    selection = json.loads(selection_summary_path.read_text(encoding="utf-8"))

    merged = assignments.merge(cleaned, on=["anime_id", "name"], how="left")
    labels = assignments["cluster"].to_numpy()
    prob_cols = _top_probability_columns(assignments)
    cluster_sizes = assignments["cluster"].value_counts().sort_index().rename_axis("cluster").reset_index(name="count")
    cluster_sizes["proportion"] = cluster_sizes["count"] / len(assignments)

    metrics = {
        "bic": selection["best_metrics"]["bic"],
        "aic": selection["best_metrics"]["aic"],
        "average_log_likelihood": selection["best_metrics"]["log_likelihood"],
        "silhouette_score": float(silhouette_score(x, labels)) if len(np.unique(labels)) > 1 else None,
        "davies_bouldin_index": float(davies_bouldin_score(x, labels)) if len(np.unique(labels)) > 1 else None,
        "calinski_harabasz_score": float(calinski_harabasz_score(x, labels)) if len(np.unique(labels)) > 1 else None,
        "cluster_size_distribution": cluster_sizes.to_dict(orient="records"),
        "min_cluster_size": int(cluster_sizes["count"].min()),
        "max_cluster_size": int(cluster_sizes["count"].max()),
        "mean_cluster_size": float(cluster_sizes["count"].mean()),
        "mean_max_responsibility": float(assignments["max_probability"].mean()),
        "median_max_responsibility": float(assignments["max_probability"].median()),
        "assignment_entropy_mean": float(assignments["assignment_entropy"].mean()),
        "assignment_entropy_median": float(assignments["assignment_entropy"].median()),
        "assignment_entropy_max": float(assignments["assignment_entropy"].max()),
    }

    numeric_columns = [
        column for column in ["score", "episodes", "duration_minutes", "members", "favorites", "scored_by", "popularity", "rank"]
        if column in merged.columns
    ]
    global_means = merged[numeric_columns].mean(numeric_only=True)
    global_stds = merged[numeric_columns].std(numeric_only=True).replace(0, 1)

    profile_rows: list[dict] = []
    report_sections = ["# Cluster Profile Summary"]
    for cluster_id in sorted(assignments["cluster"].unique()):
        part = merged[merged["cluster"] == cluster_id].copy()
        numeric_means = part[numeric_columns].mean(numeric_only=True) if numeric_columns else pd.Series(dtype=float)
        z_scores = ((numeric_means - global_means) / global_stds).sort_values(ascending=False) if numeric_columns else pd.Series(dtype=float)
        top_genres = (
            part["genres"].fillna("missing").astype(str).str.split("|").explode().value_counts().head(5).index.tolist()
            if "genres" in part.columns
            else []
        )
        top_types = part["type"].astype(str).value_counts().head(3).index.tolist() if "type" in part.columns else []
        representative = part.sort_values("max_probability", ascending=False)["name"].head(3).tolist()
        ambiguous = part.sort_values("assignment_entropy", ascending=False)["name"].head(3).tolist()
        interpretation_label = ", ".join(top_genres[:2] + top_types[:1]) if (top_genres or top_types) else f"cluster {cluster_id}"
        profile_rows.append(
            {
                "cluster": int(cluster_id),
                "size": int(len(part)),
                "proportion": float(len(part) / len(merged)),
                "top_genres": ", ".join(top_genres),
                "top_types": ", ".join(top_types),
                "top_numeric_features": ", ".join(z_scores.head(5).index.tolist()),
                "interpretation_label": interpretation_label,
                "representative_anime": ", ".join(representative),
                "ambiguous_anime": ", ".join(ambiguous),
            }
        )
        report_sections.extend(
            [
                f"## Cluster {cluster_id}",
                f"- Size: {len(part)} ({len(part) / len(merged):.2%})",
                f"- Interpretation label: {interpretation_label}",
                f"- Representative anime: {', '.join(representative) if representative else 'N/A'}",
                f"- Ambiguous anime: {', '.join(ambiguous) if ambiguous else 'N/A'}",
                f"- Dominant genres: {', '.join(top_genres) if top_genres else 'N/A'}",
                f"- Strongest numeric features: {', '.join(z_scores.head(5).index.tolist()) if len(z_scores) else 'N/A'}",
            ]
        )

    report_sections.extend(
        [
            "## Limitations",
            "- Gaussian assumptions may not perfectly fit mixed metadata features.",
            "- Sparse one-hot and multi-hot features reduce Gaussian faithfulness.",
            "- Cluster meaning depends heavily on preprocessing choices.",
            "- Unsupervised metrics do not fully capture semantic usefulness.",
        ]
    )

    write_json(metrics, metrics_path)
    write_dataframe(cluster_sizes, cluster_sizes_path)
    write_dataframe(pd.DataFrame(profile_rows), profile_csv_path)
    profile_md_path.write_text("\n".join(report_sections), encoding="utf-8")
    logger.info("Saved final evaluation outputs to %s", metrics_path)
    return {
        "metrics": str(metrics_path),
        "cluster_sizes": str(cluster_sizes_path),
        "profile_csv": str(profile_csv_path),
        "profile_md": str(profile_md_path),
    }
