"""Classical ML models for drug discovery benchmarking."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from ..utils.benchmarking import BaseModel
from ..utils.config import ExperimentConfig, RANDOM_SEED

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """Random Forest baseline."""

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        params = {
            "n_estimators": config.hyperparams.get("n_estimators", 100),
            "max_depth": config.hyperparams.get("max_depth", None),
            "random_state": config.random_seed,
            "n_jobs": -1,
        }
        if config.hyperparams.get("task_type", "classification") == "regression":
            self.model = RandomForestRegressor(**params)
        else:
            self.model = RandomForestClassifier(**params)

    @property
    def name(self) -> str:
        return "Random Forest"

    @property
    def n_parameters(self) -> int | None:
        if self.is_fitted:
            return sum(t.tree_.node_count for t in self.model.estimators_)
        return None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None


class XGBoostModel(BaseModel):
    """XGBoost baseline."""

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        params = {
            "n_estimators": config.hyperparams.get("n_estimators", 100),
            "max_depth": config.hyperparams.get("max_depth", 6),
            "learning_rate": config.hyperparams.get("xgb_lr", 0.1),
            "random_state": config.random_seed,
            "n_jobs": -1,
            "verbosity": 0,
        }
        if config.hyperparams.get("task_type", "classification") == "regression":
            self.model = XGBRegressor(**params)
        else:
            self.model = XGBClassifier(**params, eval_metric="logloss")

    @property
    def name(self) -> str:
        return "XGBoost"

    @property
    def n_parameters(self) -> int | None:
        if self.is_fitted:
            # Approximate: number of leaves across all trees
            booster = self.model.get_booster()
            trees_df = booster.trees_to_dataframe()
            return len(trees_df[trees_df["Feature"] == "Leaf"])
        return None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None


class SVMModel(BaseModel):
    """SVM baseline with RBF kernel."""

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        params = {
            "C": config.hyperparams.get("C", 1.0),
            "kernel": config.hyperparams.get("kernel", "rbf"),
            "random_state": config.random_seed,
        }
        if config.hyperparams.get("task_type", "classification") == "regression":
            self.model = SVR(**{k: v for k, v in params.items() if k != "random_state"})
        else:
            self.model = SVC(**params, probability=True)

    @property
    def name(self) -> str:
        return "SVM (RBF)"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None
