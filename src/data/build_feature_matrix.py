from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler

from src.utils.io import read_dataframe, write_dataframe, write_json


def _build_genre_features(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, list[str]]:
    mlb = MultiLabelBinarizer()
    matrix = mlb.fit_transform(df[column])
    feature_names = [f"{column}__{name}" for name in mlb.classes_]
    return pd.DataFrame(matrix, columns=feature_names, index=df.index), feature_names


def run(
    transformed_path: Path,
    features_config: dict,
    processed_dir: Path,
    checkpoints_dir: Path,
    logger: logging.Logger,
    overwrite: bool = False,
) -> dict[str, str]:
    table_path = processed_dir / "feature_ready_anime.parquet"
    matrix_path = checkpoints_dir / "feature_matrix.npy"
    names_path = checkpoints_dir / "feature_names.json"
    preprocessor_path = checkpoints_dir / "feature_preprocessor.joblib"
    metadata_path = checkpoints_dir / "feature_matrix_metadata.json"
    if table_path.exists() and matrix_path.exists() and not overwrite:
        logger.info("Skipping feature build; outputs already exist")
        return {
            "table": str(table_path),
            "matrix": str(matrix_path),
            "feature_names": str(names_path),
            "preprocessor": str(preprocessor_path),
            "metadata": str(metadata_path),
        }

    df = read_dataframe(transformed_path)
    numeric_columns = [column for column in features_config["numeric_columns"] if column in df.columns]
    categorical_columns = [column for column in features_config["categorical_columns"] if column in df.columns]
    multilabel_column = features_config["multi_label_columns"][0]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
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
    )

    base_matrix = preprocessor.fit_transform(df)
    base_feature_names = list(preprocessor.get_feature_names_out())
    genre_df, genre_feature_names = _build_genre_features(df, multilabel_column)

    matrix = np.hstack([base_matrix, genre_df.to_numpy(dtype=float)])
    feature_names = base_feature_names + genre_feature_names

    feature_table = pd.DataFrame(matrix, columns=feature_names)
    feature_table.insert(0, "anime_id", df["anime_id"].values)
    feature_table.insert(1, "name", df["name"].values)

    write_dataframe(feature_table, table_path)
    np.save(matrix_path, matrix)
    joblib.dump(preprocessor, preprocessor_path)
    write_json(feature_names, names_path)
    write_json(
        {
            "n_rows": int(matrix.shape[0]),
            "n_features": int(matrix.shape[1]),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "multi_label_columns": features_config["multi_label_columns"],
        },
        metadata_path,
    )
    logger.info("Saved feature matrix with shape=%s to %s", matrix.shape, matrix_path)
    return {
        "table": str(table_path),
        "matrix": str(matrix_path),
        "feature_names": str(names_path),
        "preprocessor": str(preprocessor_path),
        "metadata": str(metadata_path),
    }
