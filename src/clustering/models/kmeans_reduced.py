from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from src.clustering.base import ClusteringModelAdapter


class KMeansReducedModelAdapter(ClusteringModelAdapter):
    model_name = "kmeans_reduced"

    def selection_candidates(self, model_config: dict) -> list[dict[str, Any]]:
        selection_cfg = model_config["selection"]
        training_cfg = model_config["training"]
        return [{**training_cfg, "k": k} for k in selection_cfg["candidate_k"]]

    def fit_candidate(self, x: np.ndarray, params: dict[str, Any], random_seed: int) -> tuple[KMeans, dict[str, Any]]:
        model = self.fit_final(x, params, random_seed)
        labels = model.labels_
        cluster_sizes = np.bincount(labels, minlength=model.n_clusters)
        metrics: dict[str, Any] = {
            "k": int(params["k"]),
            "inertia": float(model.inertia_),
            "n_iter": int(model.n_iter_),
            "smallest_cluster_proportion": float(cluster_sizes.min() / len(x)),
            "largest_cluster_proportion": float(cluster_sizes.max() / len(x)),
            "cluster_size_distribution": cluster_sizes.tolist(),
            "degenerate_component": bool(np.any(cluster_sizes == 0)),
            "init": params["init"],
            "algorithm": params["algorithm"],
        }
        if len(np.unique(labels)) > 1:
            sample_size = min(5000, len(x))
            metrics["silhouette"] = float(silhouette_score(x, labels, sample_size=sample_size, random_state=random_seed))
            metrics["davies_bouldin"] = float(davies_bouldin_score(x, labels))
            metrics["calinski_harabasz"] = float(calinski_harabasz_score(x, labels))
        else:
            metrics["silhouette"] = None
            metrics["davies_bouldin"] = None
            metrics["calinski_harabasz"] = None
        return model, metrics

    def pick_best_params(self, results: pd.DataFrame, model_config: dict) -> dict[str, Any]:
        best = results.iloc[0].to_dict()
        return {"k": int(best["k"])}

    def fit_final(self, x: np.ndarray, params: dict[str, Any], random_seed: int) -> KMeans:
        model = KMeans(
            n_clusters=int(params["k"]),
            init=params["init"],
            n_init=int(params["n_init"]),
            max_iter=int(params["max_iter"]),
            tol=float(params["tol"]),
            algorithm=params["algorithm"],
            random_state=random_seed,
        )
        model.fit(x)
        return model

    def predict(self, model: KMeans, x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        return model.predict(x), None

    def diagnostics(self, model: KMeans) -> dict[str, Any]:
        cluster_sizes = np.bincount(model.labels_, minlength=model.n_clusters)
        return {
            "converged": bool(model.n_iter_ < model.max_iter),
            "n_iter": int(model.n_iter_),
            "inertia": float(model.inertia_),
            "weights": (cluster_sizes / cluster_sizes.sum()).tolist(),
            "means": np.asarray(model.cluster_centers_).tolist(),
            "component_count": int(model.n_clusters),
            "algorithm": model.algorithm,
            "init": model.init,
        }
