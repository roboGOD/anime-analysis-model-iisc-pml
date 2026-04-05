from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.modeling.artifacts import full_matrix_npy_path, matrix_npy_path, matrix_table_path
from src.utils.io import write_dataframe, write_json


def _resolve_n_components(reduction_config: dict) -> tuple[int | float, str]:
    fixed = reduction_config.get("n_components")
    target = reduction_config.get("explained_variance_target")
    selector = reduction_config.get("component_selection", "fixed")
    if selector == "explained_variance" and target is not None:
        return float(target), "explained_variance_target"
    if selector == "fixed" and fixed is not None:
        return int(fixed), "n_components"
    if fixed is not None:
        return int(fixed), "n_components"
    if target is not None:
        return float(target), "explained_variance_target"
    raise ValueError("Reduction config must set n_components or explained_variance_target")


def _plot_explained_variance(
    explained_ratio: np.ndarray,
    cumulative_ratio: np.ndarray,
    plots_dir: Path,
    logger: logging.Logger,
) -> dict[str, str]:
    explained_path = plots_dir / "gmm_reduced_pca_explained_variance.png"
    cumulative_path = plots_dir / "gmm_reduced_pca_cumulative_variance.png"

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(np.arange(1, len(explained_ratio) + 1), explained_ratio, color="#4e79a7")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA Explained Variance by Component")
    fig.tight_layout()
    fig.savefig(explained_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(np.arange(1, len(cumulative_ratio) + 1), cumulative_ratio, marker="o", color="#e15759")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_ylim(0, 1.02)
    ax.set_title("PCA Cumulative Explained Variance")
    fig.tight_layout()
    fig.savefig(cumulative_path, dpi=150)
    plt.close(fig)

    logger.info("Saved PCA variance plots to %s", plots_dir)
    return {
        "explained_variance_plot": str(explained_path),
        "cumulative_variance_plot": str(cumulative_path),
    }


def _plot_projection(x_reduced: np.ndarray, plots_dir: Path) -> str | None:
    if x_reduced.shape[1] < 2:
        return None
    projection_path = plots_dir / "gmm_reduced_pca_projection_raw.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x_reduced[:, 0], x_reduced[:, 1], s=12, alpha=0.55, color="#59a14f")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Projection Before Clustering")
    fig.tight_layout()
    fig.savefig(projection_path, dpi=150)
    plt.close(fig)
    return str(projection_path)


def _plot_component_loadings(loadings: pd.DataFrame, plots_dir: Path, top_n: int) -> list[str]:
    output_paths: list[str] = []
    for component in loadings["component"].unique()[: min(5, loadings["component"].nunique())]:
        part = loadings[loadings["component"] == component].copy()
        if part.empty:
            continue
        strongest = pd.concat([part.nlargest(top_n, "loading"), part.nsmallest(top_n, "loading")], ignore_index=True)
        strongest = strongest.drop_duplicates(subset=["feature_name"]).sort_values("loading")
        path = plots_dir / f"gmm_reduced_pca_component_loadings_{component.lower()}.png"
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = np.where(strongest["loading"] >= 0, "#4e79a7", "#e15759")
        ax.barh(strongest["feature_name"], strongest["loading"], color=colors)
        ax.set_title(f"{component} Top PCA Loadings")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        output_paths.append(str(path))
    return output_paths


def run(
    processed_dir: Path,
    checkpoints_dir: Path,
    reports_dir: Path,
    plots_dir: Path,
    model_config: dict,
    logger: logging.Logger,
    overwrite: bool = False,
) -> dict[str, str]:
    model_name = model_config["model"]["name"]
    source_matrix_path = full_matrix_npy_path(processed_dir, model_config)
    feature_names_path = checkpoints_dir / "feature_names.json"
    matrix_path = matrix_npy_path(processed_dir, model_config, model_name)
    table_path = matrix_table_path(processed_dir, model_config, model_name)
    pca_model_path = checkpoints_dir / f"{model_name}_pca_model.joblib"
    metadata_path = checkpoints_dir / f"{model_name}_pca_metadata.json"
    loadings_path = checkpoints_dir / f"{model_name}_pca_component_loadings.parquet"
    component_names_path = checkpoints_dir / f"{model_name}_feature_names.json"
    summary_path = reports_dir / f"{model_name}_full_space_summary_before_reduction.json"
    if all(
        path.exists()
        for path in [matrix_path, table_path, pca_model_path, metadata_path, loadings_path, component_names_path, summary_path]
    ) and not overwrite:
        logger.info("Skipping feature reduction; outputs already exist")
        return {
            "matrix": str(matrix_path),
            "matrix_table": str(table_path),
            "pca_model": str(pca_model_path),
            "pca_metadata": str(metadata_path),
            "pca_loadings": str(loadings_path),
            "feature_names": str(component_names_path),
            "pre_reduction_summary": str(summary_path),
        }

    x_full = np.load(source_matrix_path)
    if x_full.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, found shape={x_full.shape}")
    if not np.issubdtype(x_full.dtype, np.number):
        raise ValueError("Full feature matrix must be numeric")
    if not np.isfinite(x_full).all():
        raise ValueError("Full feature matrix contains NaN or infinite values")

    reduction_config = model_config["reduction"]
    pca_config = reduction_config.get("pca", {})
    n_components, selected_by = _resolve_n_components(reduction_config)
    feature_names = []
    if feature_names_path.exists():
        feature_names = json.loads(feature_names_path.read_text(encoding="utf-8"))
    if not feature_names:
        feature_names = [f"feature_{index}" for index in range(x_full.shape[1])]

    sparsity = float(np.mean(x_full == 0))
    write_json(
        {
            "source_matrix_path": str(source_matrix_path),
            "row_count": int(x_full.shape[0]),
            "feature_count": int(x_full.shape[1]),
            "dtype": str(x_full.dtype),
            "sparsity_ratio": sparsity,
            "selected_component_rule": selected_by,
        },
        summary_path,
    )

    pca = PCA(
        n_components=n_components,
        svd_solver=pca_config.get("svd_solver", "auto"),
        whiten=bool(pca_config.get("whiten", False)),
        random_state=pca_config.get("random_seed"),
    )
    x_reduced = pca.fit_transform(x_full)
    component_names = [f"pc_{index + 1:03d}" for index in range(x_reduced.shape[1])]
    x_reduced_df = pd.DataFrame(x_reduced, columns=component_names)

    explained_ratio = pca.explained_variance_ratio_
    cumulative_ratio = np.cumsum(explained_ratio)
    loadings = pd.DataFrame(pca.components_.T, columns=component_names, index=feature_names)
    loadings_long = (
        loadings.rename_axis("feature_name")
        .reset_index()
        .melt(id_vars="feature_name", var_name="component", value_name="loading")
    )
    top_n = int(reduction_config.get("top_loading_features", 10))

    np.save(matrix_path, x_reduced)
    write_dataframe(x_reduced_df, table_path)
    joblib.dump(pca, pca_model_path)
    write_dataframe(loadings_long, loadings_path)
    write_json(component_names, component_names_path)
    write_json(
        {
            "model_name": model_name,
            "source_matrix_path": str(source_matrix_path),
            "reduced_matrix_path": str(matrix_path),
            "selection_rule": selected_by,
            "requested_n_components": reduction_config.get("n_components"),
            "requested_explained_variance_target": reduction_config.get("explained_variance_target"),
            "retained_component_count": int(x_reduced.shape[1]),
            "original_feature_count": int(x_full.shape[1]),
            "row_count": int(x_full.shape[0]),
            "explained_variance_ratio": explained_ratio.tolist(),
            "cumulative_explained_variance": cumulative_ratio.tolist(),
            "total_explained_variance": float(cumulative_ratio[-1]) if len(cumulative_ratio) else 0.0,
            "whiten": bool(pca_config.get("whiten", False)),
            "svd_solver": pca_config.get("svd_solver", "auto"),
            "top_loading_features_per_component": top_n,
        },
        metadata_path,
    )

    outputs = {
        "matrix": str(matrix_path),
        "matrix_table": str(table_path),
        "pca_model": str(pca_model_path),
        "pca_metadata": str(metadata_path),
        "pca_loadings": str(loadings_path),
        "feature_names": str(component_names_path),
        "pre_reduction_summary": str(summary_path),
    }
    outputs.update(_plot_explained_variance(explained_ratio, cumulative_ratio, plots_dir, logger))
    projection_path = _plot_projection(x_reduced, plots_dir)
    if projection_path:
        outputs["projection_plot"] = projection_path
    component_plot_paths = _plot_component_loadings(loadings_long, plots_dir, top_n=top_n)
    for index, path in enumerate(component_plot_paths, start=1):
        outputs[f"component_loading_plot_{index}"] = path

    logger.info("Saved reduced feature matrix with shape=%s to %s", x_reduced.shape, matrix_path)
    return outputs
