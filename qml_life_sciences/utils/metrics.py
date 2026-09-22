"""Unified metric computation for classical and quantum model evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


@dataclass
class ClassificationMetrics:
    """Stores classification metrics for a single evaluation."""

    accuracy: float
    f1_macro: float
    f1_weighted: float
    precision_macro: float
    recall_macro: float
    auc_roc: float | None  # None if not computable (e.g. multiclass without probas)
    training_time_seconds: float
    inference_time_seconds: float
    n_train_samples: int
    n_test_samples: int
    n_parameters: int | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "accuracy": self.accuracy,
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "auc_roc": self.auc_roc,
            "training_time_s": self.training_time_seconds,
            "inference_time_s": self.inference_time_seconds,
            "n_train": self.n_train_samples,
            "n_test": self.n_test_samples,
            "n_parameters": self.n_parameters,
        }
        d.update(self.extra)
        return d


@dataclass
class RegressionMetrics:
    """Stores regression metrics for a single evaluation."""

    rmse: float
    mae: float
    r2: float
    training_time_seconds: float
    inference_time_seconds: float
    n_train_samples: int
    n_test_samples: int
    n_parameters: int | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "training_time_s": self.training_time_seconds,
            "inference_time_s": self.inference_time_seconds,
            "n_train": self.n_train_samples,
            "n_test": self.n_test_samples,
            "n_parameters": self.n_parameters,
        }
        d.update(self.extra)
        return d


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    training_time: float,
    inference_time: float,
    n_train: int,
    n_parameters: int | None = None,
) -> ClassificationMetrics:
    """Compute all classification metrics from predictions.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities (n_samples, n_classes) or None.
        training_time: Wall-clock training time in seconds.
        inference_time: Wall-clock inference time in seconds.
        n_train: Number of training samples used.
        n_parameters: Number of trainable parameters (if known).

    Returns:
        ClassificationMetrics with all computed values.
    """
    auc = None
    if y_prob is not None:
        try:
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                # Binary: use probability of positive class
                prob_col = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                auc = roc_auc_score(y_true, prob_col)
            else:
                auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except (ValueError, IndexError):
            auc = None

    return ClassificationMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        f1_macro=f1_score(y_true, y_pred, average="macro", zero_division=0),
        f1_weighted=f1_score(y_true, y_pred, average="weighted", zero_division=0),
        precision_macro=precision_score(y_true, y_pred, average="macro", zero_division=0),
        recall_macro=recall_score(y_true, y_pred, average="macro", zero_division=0),
        auc_roc=auc,
        training_time_seconds=training_time,
        inference_time_seconds=inference_time,
        n_train_samples=n_train,
        n_test_samples=len(y_true),
        n_parameters=n_parameters,
    )


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    training_time: float,
    inference_time: float,
    n_train: int,
    n_parameters: int | None = None,
) -> RegressionMetrics:
    """Compute all regression metrics from predictions."""
    return RegressionMetrics(
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        mae=float(mean_absolute_error(y_true, y_pred)),
        r2=float(r2_score(y_true, y_pred)),
        training_time_seconds=training_time,
        inference_time_seconds=inference_time,
        n_train_samples=n_train,
        n_test_samples=len(y_true),
        n_parameters=n_parameters,
    )


class Timer:
    """Simple context manager for timing code blocks."""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
