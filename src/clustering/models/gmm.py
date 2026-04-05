from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from src.clustering.base import ClusteringModelAdapter


class GMMModelAdapter(ClusteringModelAdapter):
    model_name = "gmm"

    def selection_candidates(self, model_config: dict) -> list[dict[str, Any]]:
        selection_cfg = model_config["selection"]
        training_cfg = model_config["training"]
        return [
            {
                "k": k,
                "covariance_type": covariance_type,
                **training_cfg,
            }
            for k in selection_cfg["candidate_k"]
            for covariance_type in selection_cfg["covariance_types"]
        ]

    def fit_candidate(self, x: np.ndarray, params: dict[str, Any], random_seed: int) -> tuple[GaussianMixture, dict[str, Any]]:
        model = GaussianMixture(
            n_components=int(params["k"]),
            covariance_type=params["covariance_type"],
            max_iter=int(params["max_iter"]),
            tol=float(params["tol"]),
            reg_covar=float(params["reg_covar"]),
            n_init=int(params["n_init"]),
            init_params=params["init_params"],
            random_state=random_seed,
        )
        labels = model.fit_predict(x)
        probabilities = model.predict_proba(x)
        cluster_sizes = np.bincount(labels, minlength=model.n_components)
        metrics: dict[str, Any] = {
            "k": int(params["k"]),
            "covariance_type": params["covariance_type"],
            "bic": float(model.bic(x)),
            "aic": float(model.aic(x)),
            "log_likelihood": float(model.score(x)),
            "converged": bool(model.converged_),
            "n_iter": int(model.n_iter_),
            "lower_bound": float(model.lower_bound_),
            "smallest_cluster_proportion": float(cluster_sizes.min() / len(x)),
            "largest_cluster_proportion": float(cluster_sizes.max() / len(x)),
            "cluster_size_distribution": cluster_sizes.tolist(),
            "tiny_cluster_count": int(np.sum(cluster_sizes / len(x) < 0.01)),
            "degenerate_component": bool(np.any(model.weights_ < 1e-4)),
            "mean_max_responsibility": float(probabilities.max(axis=1).mean()),
        }
        if len(set(labels)) > 1:
            metrics["silhouette"] = float(
                silhouette_score(x, labels, sample_size=min(5000, len(x)), random_state=random_seed)
            )
        else:
            metrics["silhouette"] = None
        return model, metrics

    def pick_best_params(self, results: pd.DataFrame, model_config: dict) -> dict[str, Any]:
        best = results.iloc[0].to_dict()
        return {
            "k": int(best["k"]),
            "covariance_type": best["covariance_type"],
        }

    def fit_final(self, x: np.ndarray, params: dict[str, Any], random_seed: int) -> GaussianMixture:
        final_params = {**params}
        model = GaussianMixture(
            n_components=int(final_params["k"]),
            covariance_type=final_params["covariance_type"],
            max_iter=int(final_params["max_iter"]),
            tol=float(final_params["tol"]),
            reg_covar=float(final_params["reg_covar"]),
            n_init=int(final_params["n_init"]),
            init_params=final_params["init_params"],
            random_state=random_seed,
        )
        model.fit(x)
        return model

    def predict(self, model: GaussianMixture, x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        return model.predict(x), model.predict_proba(x)

    def diagnostics(self, model: GaussianMixture) -> dict[str, Any]:
        return {
            "converged": bool(model.converged_),
            "n_iter": int(model.n_iter_),
            "lower_bound": float(model.lower_bound_),
            "weights": model.weights_.tolist(),
            "means": model.means_.tolist(),
            "component_count": int(model.n_components),
            "covariance_type": model.covariance_type,
            "covariances": np.asarray(model.covariances_).tolist(),
        }
