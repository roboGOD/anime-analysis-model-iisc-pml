from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.io import read_dataframe, write_dataframe, write_json


def run(
    transformed_path: Path,
    features_config: dict,
    processed_dir: Path,
    checkpoints_dir: Path,
    metrics_dir: Path,
    reports_dir: Path,
    logger: logging.Logger,
    overwrite: bool = False,
) -> dict[str, str]:
    matrix_npy_path = processed_dir / "X_gmm.npy"
    matrix_table_path = processed_dir / "X_gmm.parquet"
    names_path = checkpoints_dir / "feature_names.json"
    row_mapping_path = checkpoints_dir / "row_mapping.parquet"
    metrics_path = metrics_dir / "data_quality_metrics.json"
    interpretation_slice_path = reports_dir / "cluster_interpretation_metadata.parquet"
    if matrix_npy_path.exists() and matrix_table_path.exists() and names_path.exists() and not overwrite:
        logger.info("Skipping feature matrix build; outputs already exist")
        return {
            "matrix": str(matrix_npy_path),
            "matrix_table": str(matrix_table_path),
            "feature_names": str(names_path),
            "row_mapping": str(row_mapping_path),
            "metrics": str(metrics_path),
        }

    df = read_dataframe(transformed_path).reset_index(drop=True)
    id_columns = [column for column in ["anime_id", "name"] if column in df.columns]
    numeric_columns = []
    for column in features_config["numeric_columns"]:
        if f"{column}_log1p" in df.columns:
            numeric_columns.append(f"{column}_log1p")
        elif column in df.columns:
            numeric_columns.append(column)
    if "duration_minutes" in df.columns and "duration_minutes" not in numeric_columns:
        numeric_columns.append("duration_minutes")

    categorical_columns = [column for column in features_config["categorical_columns"] if column in df.columns]
    tag_columns = sorted(
        [column for column in df.columns if any(column.startswith(f"{base}__") for base in features_config["multi_label_columns"])]
    )

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown=features_config["encoder"]["handle_unknown"],
                    min_frequency=features_config["encoder"]["one_hot_min_frequency"],
                    sparse_output=False,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    base_matrix = preprocessor.fit_transform(df)
    base_feature_names = preprocessor.get_feature_names_out().tolist()
    tag_matrix = df[tag_columns].to_numpy(dtype=float) if tag_columns else np.empty((len(df), 0))

    x = np.hstack([base_matrix, tag_matrix]).astype(float)
    if np.isnan(x).any():
        raise ValueError("Final feature matrix contains NaN values")

    feature_names = base_feature_names + tag_columns
    x_table = pd.DataFrame(x, columns=feature_names)
    row_mapping = df[id_columns].copy()
    row_mapping["row_index"] = np.arange(len(df))
    interpretation_columns = id_columns + [column for column in ["type", "source", "rating", "genres", "score"] if column in df.columns]

    np.save(matrix_npy_path, x)
    write_dataframe(x_table, matrix_table_path)
    write_json(feature_names, names_path)
    write_dataframe(row_mapping, row_mapping_path)
    write_dataframe(df[interpretation_columns], interpretation_slice_path)

    sparse_ratio = float((x_table.eq(0).sum().sum()) / (x_table.shape[0] * max(x_table.shape[1], 1)))
    metrics = {
        "raw_row_count": int(len(df)),
        "cleaned_row_count": int(len(df)),
        "rows_dropped": 0,
        "duplicate_count": int(df.duplicated(subset=["anime_id"]).sum()) if "anime_id" in df.columns else 0,
        "missing_value_count_per_selected_feature": {
            column: int(df[column].isna().sum()) for column in numeric_columns + categorical_columns
        },
        "final_feature_dimensionality": int(x.shape[1]),
        "proportion_of_sparse_binary_features": sparse_ratio,
        "numeric_feature_count": int(len(base_feature_names) - sum(name.startswith("categorical__") for name in base_feature_names)),
        "categorical_feature_count": int(sum(name.startswith("categorical__") for name in base_feature_names)),
        "tag_feature_count": int(len(tag_columns)),
        "top_frequent_genres_retained": [column.replace("genres__", "") for column in tag_columns if column.startswith("genres__")][:20],
        "rows_with_any_imputation_applied": int(
            df[numeric_columns + categorical_columns].isna().any(axis=1).sum()
        ) if numeric_columns or categorical_columns else 0,
    }
    write_json(metrics, metrics_path)
    logger.info("Saved GMM-ready matrix with shape=%s to %s", x.shape, matrix_npy_path)
    return {
        "matrix": str(matrix_npy_path),
        "matrix_table": str(matrix_table_path),
        "feature_names": str(names_path),
        "row_mapping": str(row_mapping_path),
        "metrics": str(metrics_path),
    }
