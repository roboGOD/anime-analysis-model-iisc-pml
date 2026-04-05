from __future__ import annotations

from src.clustering.base import ClusteringModelAdapter
from src.clustering.models.gmm import GMMModelAdapter
from src.clustering.models.gmm_reduced import GMMReducedModelAdapter


MODEL_REGISTRY: dict[str, type[ClusteringModelAdapter]] = {
    "gmm": GMMModelAdapter,
    "gmm_reduced": GMMReducedModelAdapter,
}


def create_model_adapter(model_name: str) -> ClusteringModelAdapter:
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported model '{model_name}'. Available models: {available}")
    return MODEL_REGISTRY[model_name]()
