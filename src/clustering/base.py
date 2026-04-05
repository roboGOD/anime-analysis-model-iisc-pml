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

    def best_row(self, results: pd.DataFrame, best_params: dict[str, Any]) -> dict[str, Any]:
        mask = pd.Series(True, index=results.index)
        for key, value in best_params.items():
            if key not in results.columns:
                continue
            mask &= results[key] == value
        if mask.any():
            return results.loc[mask].iloc[0].to_dict()
        return results.iloc[0].to_dict()
