"""Classical ML models for genomics benchmarking.

Reuses the same model classes from drug_discovery.classical since the
BaseModel interface is universal. This module adds any genomics-specific
model configurations or additional models (e.g., MLP).
"""

import numpy as np
from sklearn.neural_network import MLPClassifier

from ..utils.benchmarking import BaseModel
from ..utils.config import ExperimentConfig, RANDOM_SEED

# Re-export shared models
from ..drug_discovery.classical import (  # noqa: F401
    RandomForestModel,
    XGBoostModel,
    SVMModel,
)


class MLPModel(BaseModel):
    """Multi-Layer Perceptron baseline (specific to genomics)."""

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.model = MLPClassifier(
            hidden_layer_sizes=config.hyperparams.get("hidden_layers", (64, 32)),
            max_iter=config.hyperparams.get("mlp_max_iter", 500),
            random_state=config.random_seed,
            early_stopping=True,
            validation_fraction=0.15,
        )

    @property
    def name(self) -> str:
        return "MLP"

    @property
    def n_parameters(self) -> int | None:
        if self.is_fitted:
            return sum(w.size for w in self.model.coefs_) + sum(b.size for b in self.model.intercepts_)
        return None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        return self.model.predict_proba(X)
