from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class ClusteringModelAdapter(ABC):
    model_name: str

    @abstractmethod
    def selection_candidates(self, model_config: dict) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def fit_candidate(self, x: np.ndarray, params: dict[str, Any], random_seed: int) -> tuple[Any, dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def pick_best_params(self, results: pd.DataFrame, model_config: dict) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def fit_final(self, x: np.ndarray, params: dict[str, Any], random_seed: int) -> Any:
        raise NotImplementedError

    @abstractmethod
    def predict(self, model: Any, x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        raise NotImplementedError

    @abstractmethod
    def diagnostics(self, model: Any) -> dict[str, Any]:
        raise NotImplementedError
