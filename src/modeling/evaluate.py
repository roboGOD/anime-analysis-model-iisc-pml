from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from src.modeling.artifacts import cluster_profile_csv_path, cluster_profile_md_path
from src.utils.io import read_dataframe, write_dataframe, write_json


def _top_probability_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column.startswith("cluster_prob_")]


def _split_multilabel(series: pd.Series) -> pd.Series:
    return (
        series.fillna("missing")
        .astype(str)
        .str.split("|")
        .explode()
        .astype(str)
        .str.strip()
        .replace("", "missing")
    )


def _distribution_records(part: pd.DataFrame, merged: pd.DataFrame, column: str, top_n: int = 5) -> list[dict]:
    if column not in merged.columns or column not in part.columns:
        return []
    cluster_counts = part[column].fillna("missing").astype(str).value_counts()
    global_counts = merged[column].fillna("missing").astype(str).value_counts()
    records: list[dict] = []
    for value, count in cluster_counts.head(top_n).items():
        cluster_share = float(count / len(part))
        global_share = float(global_counts.get(value, 0) / len(merged))
        lift = float(cluster_share / global_share) if global_share > 0 else None
        records.append(
            {
                "value": value,
                "count": int(count),
                "cluster_share": cluster_share,
                "global_share": global_share,
                "lift_vs_global": lift,
            }
        )
    return records


def _multilabel_distribution_records(part: pd.DataFrame, merged: pd.DataFrame, column: str, top_n: int = 8) -> list[dict]:
    if column not in merged.columns or column not in part.columns:
        return []
    cluster_counts = _split_multilabel(part[column]).value_counts()
    global_counts = _split_multilabel(merged[column]).value_counts()
    records: list[dict] = []
    for value, count in cluster_counts.head(top_n).items():
        cluster_share = float(count / len(part))
        global_share = float(global_counts.get(value, 0) / len(merged))
        lift = float(cluster_share / global_share) if global_share > 0 else None
        records.append(
            {
                "value": value,
                "count": int(count),
                "cluster_share": cluster_share,
                "global_share": global_share,
                "lift_vs_global": lift,
            }
        )
    return records


def _overindexed_values(records: list[dict], min_cluster_share: float = 0.1, top_n: int = 3) -> list[str]:
    ranked = sorted(
        [
            record
            for record in records
            if record["cluster_share"] >= min_cluster_share and record.get("lift_vs_global") is not None
        ],
        key=lambda item: (item["lift_vs_global"], item["cluster_share"]),
        reverse=True,
    )
    return [record["value"] for record in ranked[:top_n]]


def _numeric_distribution_summary(part: pd.DataFrame, column: str, global_means: pd.Series, global_stds: pd.Series) -> dict | None:
    if column not in part.columns:
        return None
    values = pd.to_numeric(part[column], errors="coerce").dropna()
    if values.empty:
        return None
    mean_value = float(values.mean())
    return {
        "mean": mean_value,
        "median": float(values.median()),
        "p25": float(values.quantile(0.25)),
        "p75": float(values.quantile(0.75)),
        "std": float(values.std()) if len(values) > 1 else 0.0,
        "z_score_vs_global": float((mean_value - global_means[column]) / global_stds[column]) if column in global_means.index else None,
    }


def _distribution_line(records: list[dict], empty_label: str = "N/A") -> str:
    if not records:
        return empty_label
    return ", ".join(
        f"{record['value']} ({record['cluster_share']:.1%}, lift {record['lift_vs_global']:.2f})"
        if record.get("lift_vs_global") is not None
        else f"{record['value']} ({record['cluster_share']:.1%})"
        for record in records
    )


def run(
    model_name: str,
    assignments_path: Path,
    matrix_path: Path,
    selection_summary_path: Path,
    cleaned_metadata_path: Path,
    metrics_dir: Path,
    reports_dir: Path,
    logger: logging.Logger,
    overwrite: bool = False,
) -> dict[str, str]:
    metrics_path = metrics_dir / f"final_{model_name}_metrics.json"
    profile_csv_path = cluster_profile_csv_path(reports_dir, model_name)
    profile_md_path = cluster_profile_md_path(reports_dir, model_name)
    cluster_sizes_path = metrics_dir / f"{model_name}_cluster_sizes.csv"
    distribution_json_path = reports_dir / f"{model_name}_cluster_distribution_analysis.json"
    distribution_md_path = reports_dir / f"{model_name}_cluster_distribution_analysis.md"
    if (
        metrics_path.exists()
        and profile_csv_path.exists()
        and profile_md_path.exists()
        and distribution_json_path.exists()
        and distribution_md_path.exists()
        and not overwrite
    ):
        logger.info("Skipping evaluation; outputs already exist")
        return {
            "metrics": str(metrics_path),
            "cluster_sizes": str(cluster_sizes_path),
            "profile_csv": str(profile_csv_path),
            "profile_md": str(profile_md_path),
            "distribution_json": str(distribution_json_path),
            "distribution_md": str(distribution_md_path),
        }

    assignments = read_dataframe(assignments_path)
    cleaned = read_dataframe(cleaned_metadata_path)
    x = np.load(matrix_path)
    selection = json.loads(selection_summary_path.read_text(encoding="utf-8"))
    best_selection_metrics = selection["best_metrics"]

    merged = assignments.merge(cleaned, on=["anime_id", "name"], how="left")
    labels = assignments["cluster"].to_numpy()
    cluster_sizes = assignments["cluster"].value_counts().sort_index().rename_axis("cluster").reset_index(name="count")
    cluster_sizes["proportion"] = cluster_sizes["count"] / len(assignments)

    metrics = {
        "selection_metric_primary": selection.get("selection_metric_primary"),
        "selection_metric_secondary": selection.get("selection_metric_secondary"),
        "selection_metric_tertiary": selection.get("selection_metric_tertiary"),
        "best_selection_metrics": best_selection_metrics,
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
    if "bic" in best_selection_metrics:
        metrics["bic"] = best_selection_metrics["bic"]
    if "aic" in best_selection_metrics:
        metrics["aic"] = best_selection_metrics["aic"]
    if "log_likelihood" in best_selection_metrics:
        metrics["average_log_likelihood"] = best_selection_metrics["log_likelihood"]
    if "inertia" in best_selection_metrics:
        metrics["inertia"] = best_selection_metrics["inertia"]

    numeric_columns = [
        column for column in ["score", "episodes", "duration_minutes", "members", "favorites", "scored_by", "popularity", "rank"]
        if column in merged.columns
    ]
    global_means = merged[numeric_columns].mean(numeric_only=True)
    global_stds = merged[numeric_columns].std(numeric_only=True).replace(0, 1)
    categorical_columns = [column for column in ["type", "source", "rating", "status", "premiered_season"] if column in merged.columns]

    profile_rows: list[dict] = []
    report_sections = ["# Cluster Profile Summary"]
    distribution_analysis: list[dict] = []
    distribution_sections = ["# Cluster Distribution Analysis"]
    top_type_shares: list[float] = []
    top_source_shares: list[float] = []
    for cluster_id in sorted(assignments["cluster"].unique()):
        part = merged[merged["cluster"] == cluster_id].copy()
        numeric_means = part[numeric_columns].mean(numeric_only=True) if numeric_columns else pd.Series(dtype=float)
        z_scores = ((numeric_means - global_means) / global_stds).sort_values(ascending=False) if numeric_columns else pd.Series(dtype=float)
        genre_distribution = _multilabel_distribution_records(part, merged, "genres")
        top_genres = [record["value"] for record in genre_distribution[:5]]
        type_distribution = _distribution_records(part, merged, "type")
        source_distribution = _distribution_records(part, merged, "source")
        rating_distribution = _distribution_records(part, merged, "rating")
        status_distribution = _distribution_records(part, merged, "status")
        season_distribution = _distribution_records(part, merged, "premiered_season")
        top_types = [record["value"] for record in type_distribution[:3]]
        representative = part.sort_values("max_probability", ascending=False)["name"].head(3).tolist()
        ambiguous = part.sort_values("assignment_entropy", ascending=False)["name"].head(3).tolist()
        overindexed_genres = _overindexed_values(genre_distribution, min_cluster_share=0.08, top_n=2)
        overindexed_types = _overindexed_values(type_distribution, min_cluster_share=0.15, top_n=1)
        overindexed_sources = _overindexed_values(source_distribution, min_cluster_share=0.12, top_n=1)
        top_numeric_features = z_scores.abs().sort_values(ascending=False).head(4).index.tolist()
        interpretation_bits = overindexed_genres + overindexed_types + overindexed_sources
        interpretation_label = ", ".join(interpretation_bits[:4]) if interpretation_bits else ", ".join(top_genres[:2] + top_types[:1]) if (top_genres or top_types) else f"cluster {cluster_id}"
        numeric_distribution = {
            column: _numeric_distribution_summary(part, column, global_means, global_stds)
            for column in numeric_columns
            if _numeric_distribution_summary(part, column, global_means, global_stds) is not None
        }
        dominant_type_share = type_distribution[0]["cluster_share"] if type_distribution else 0.0
        dominant_source_share = source_distribution[0]["cluster_share"] if source_distribution else 0.0
        top_type_shares.append(dominant_type_share)
        top_source_shares.append(dominant_source_share)
        distribution_analysis.append(
            {
                "cluster": int(cluster_id),
                "size": int(len(part)),
                "proportion": float(len(part) / len(merged)),
                "interpretation_label": interpretation_label,
                "representative_anime": representative,
                "ambiguous_anime": ambiguous,
                "dominance_signals": {
                    "top_type_share": dominant_type_share,
                    "top_source_share": dominant_source_share,
                    "format_artifact_warning": bool(dominant_type_share >= 0.8 or dominant_source_share >= 0.8),
                },
                "distributions": {
                    "genres": genre_distribution,
                    "type": type_distribution,
                    "source": source_distribution,
                    "rating": rating_distribution,
                    "status": status_distribution,
                    "premiered_season": season_distribution,
                },
                "numeric_distributions": numeric_distribution,
                "top_numeric_features_by_z_score": top_numeric_features,
            }
        )
        profile_rows.append(
            {
                "cluster": int(cluster_id),
                "size": int(len(part)),
                "proportion": float(len(part) / len(merged)),
                "top_genres": ", ".join(top_genres),
                "top_types": ", ".join(top_types),
                "top_numeric_features": ", ".join(top_numeric_features),
                "interpretation_label": interpretation_label,
                "representative_anime": ", ".join(representative),
                "ambiguous_anime": ", ".join(ambiguous),
                "top_type_share": dominant_type_share,
                "top_source_share": dominant_source_share,
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
                f"- Dominant types: {', '.join(top_types) if top_types else 'N/A'}",
                f"- Strongest numeric features: {', '.join(top_numeric_features) if top_numeric_features else 'N/A'}",
            ]
        )
        distribution_sections.extend(
            [
                f"## Cluster {cluster_id}",
                f"- Size: {len(part)} ({len(part) / len(merged):.2%})",
                f"- Interpreted as: {interpretation_label}",
                f"- Representative anime: {', '.join(representative) if representative else 'N/A'}",
                f"- Boundary anime: {', '.join(ambiguous) if ambiguous else 'N/A'}",
                f"- Genre distribution: {_distribution_line(genre_distribution)}",
                f"- Type distribution: {_distribution_line(type_distribution)}",
                f"- Source distribution: {_distribution_line(source_distribution)}",
                f"- Rating distribution: {_distribution_line(rating_distribution)}",
                f"- Status distribution: {_distribution_line(status_distribution)}",
                f"- Season distribution: {_distribution_line(season_distribution)}",
                f"- Dominant numeric shifts: {', '.join(f'{feature} ({z_scores[feature]:+.2f}z)' for feature in top_numeric_features if feature in z_scores.index) if top_numeric_features else 'N/A'}",
                f"- Format-artifact warning: {'yes' if dominant_type_share >= 0.8 or dominant_source_share >= 0.8 else 'no'}",
            ]
        )

    metrics["format_artifact_summary"] = {
        "mean_top_type_share": float(np.mean(top_type_shares)) if top_type_shares else None,
        "mean_top_source_share": float(np.mean(top_source_shares)) if top_source_shares else None,
        "clusters_with_type_share_above_80pct": int(sum(share >= 0.8 for share in top_type_shares)),
        "clusters_with_source_share_above_80pct": int(sum(share >= 0.8 for share in top_source_shares)),
    }
    if "stability_ari_mean" in best_selection_metrics:
        metrics["stability_ari_mean"] = best_selection_metrics["stability_ari_mean"]
    if "stability_nmi_mean" in best_selection_metrics:
        metrics["stability_nmi_mean"] = best_selection_metrics["stability_nmi_mean"]

    report_sections.extend(["## Limitations"])
    if model_name.startswith("gmm"):
        report_sections.extend(
            [
                "- Gaussian assumptions may not perfectly fit mixed metadata features.",
                "- Sparse one-hot and multi-hot features reduce Gaussian faithfulness.",
            ]
        )
    if model_name.startswith("kmeans"):
        report_sections.extend(
            [
                "- K-means favors roughly spherical, equally scaled clusters under Euclidean distance.",
                "- Cluster centroids in PCA space may simplify away semantically useful low-variance structure.",
            ]
        )
    report_sections.extend(
        [
            "- Cluster meaning depends heavily on preprocessing choices.",
            "- Unsupervised metrics do not fully capture semantic usefulness.",
        ]
    )

    write_json(metrics, metrics_path)
    write_dataframe(cluster_sizes, cluster_sizes_path)
    write_dataframe(pd.DataFrame(profile_rows), profile_csv_path)
    profile_md_path.write_text("\n".join(report_sections), encoding="utf-8")
    write_json(
        {
            "model_name": model_name,
            "cluster_count": int(len(distribution_analysis)),
            "feature_columns_used_for_distribution": {
                "numeric": numeric_columns,
                "categorical": categorical_columns,
                "multilabel": ["genres"] if "genres" in merged.columns else [],
            },
            "clusters": distribution_analysis,
        },
        distribution_json_path,
    )
    distribution_md_path.write_text("\n".join(distribution_sections), encoding="utf-8")
    if model_name == "gmm":
        write_dataframe(pd.DataFrame(profile_rows), reports_dir / "cluster_profile_tables.csv")
        (reports_dir / "cluster_profile_summary.md").write_text("\n".join(report_sections), encoding="utf-8")
        write_json(
            {
                "model_name": model_name,
                "cluster_count": int(len(distribution_analysis)),
                "feature_columns_used_for_distribution": {
                    "numeric": numeric_columns,
                    "categorical": categorical_columns,
                    "multilabel": ["genres"] if "genres" in merged.columns else [],
                },
                "clusters": distribution_analysis,
            },
            reports_dir / "cluster_distribution_analysis.json",
        )
        (reports_dir / "cluster_distribution_analysis.md").write_text("\n".join(distribution_sections), encoding="utf-8")
    logger.info("Saved final evaluation outputs to %s", metrics_path)
    return {
        "metrics": str(metrics_path),
        "cluster_sizes": str(cluster_sizes_path),
        "profile_csv": str(profile_csv_path),
        "profile_md": str(profile_md_path),
        "distribution_json": str(distribution_json_path),
        "distribution_md": str(distribution_md_path),
    }
