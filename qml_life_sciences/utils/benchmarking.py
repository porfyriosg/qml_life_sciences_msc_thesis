"""Unified benchmarking harness for classical vs quantum model comparison.

This module provides the central experiment runner that ensures fair comparison:
same splits, same preprocessing, same metrics, for all models.
"""

from __future__ import annotations

import csv
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .config import ExperimentConfig, LOGS_DIR, RANDOM_SEED
from .metrics import (
    ClassificationMetrics,
    RegressionMetrics,
    Timer,
    compute_classification_metrics,
    compute_regression_metrics,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base class for all models (classical and quantum)
# ---------------------------------------------------------------------------

class BaseModel(ABC):
    """Interface that both classical and quantum models must implement."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.is_fitted = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name for logging."""

    @property
    def n_parameters(self) -> int | None:
        """Number of trainable parameters. Override in subclass if known."""
        return None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted labels (classification) or values (regression)."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        """Return class probabilities. Override if model supports it."""
        return None


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Runs fair comparisons between classical and quantum models.

    Ensures identical data splits, preprocessing, and evaluation protocol
    across all models for a given use case and dataset.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        task_type: str = "classification",
    ):
        """
        Args:
            config: Base experiment configuration.
            task_type: "classification" or "regression".
        """
        self.config = config
        self.task_type = task_type
        self.results: list[dict[str, Any]] = []

    def run_cross_validation(
        self,
        model: BaseModel,
        X: np.ndarray,
        y: np.ndarray,
    ) -> list[ClassificationMetrics | RegressionMetrics]:
        """Run stratified k-fold cross-validation for a single model.

        Args:
            model: A model implementing BaseModel interface.
            X: Feature matrix (n_samples, n_features).
            y: Labels or target values.

        Returns:
            List of metric objects, one per fold.
        """
        skf = StratifiedKFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.random_seed,
        )
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            logger.info(
                "  Fold %d/%d | train=%d, test=%d",
                fold_idx + 1, self.config.n_folds, len(train_idx), len(test_idx),
            )

            with Timer() as train_timer:
                model.fit(X_train, y_train)

            with Timer() as infer_timer:
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)

            if self.task_type == "classification":
                metrics = compute_classification_metrics(
                    y_true=y_test,
                    y_pred=y_pred,
                    y_prob=y_prob,
                    training_time=train_timer.elapsed,
                    inference_time=infer_timer.elapsed,
                    n_train=len(train_idx),
                    n_parameters=model.n_parameters,
                )
            else:
                metrics = compute_regression_metrics(
                    y_true=y_test,
                    y_pred=y_pred,
                    training_time=train_timer.elapsed,
                    inference_time=infer_timer.elapsed,
                    n_train=len(train_idx),
                    n_parameters=model.n_parameters,
                )

            fold_metrics.append(metrics)
            logger.info("  -> %s", _summarize_metrics(metrics))

        return fold_metrics

    def run_learning_curve(
        self,
        model_factory: type[BaseModel],
        X: np.ndarray,
        y: np.ndarray,
        fractions: list[float] | None = None,
        model_name: str | None = None,
    ) -> dict[float, list[ClassificationMetrics | RegressionMetrics]]:
        """Run learning curve experiments across different training set sizes.

        Iterates fractions first so results are checkpointed to disk after each
        fraction completes — a crash only loses the current fraction, not all work.

        Args:
            model_factory: Class to instantiate for each run (gets fresh model).
            X: Full feature matrix.
            y: Full labels.
            fractions: Training set size fractions. Defaults to config values.
            model_name: Label used in checkpoint filenames.

        Returns:
            Dict mapping fraction -> list of fold metrics.
        """
        fractions = fractions or self.config.learning_curve_fractions
        rng = np.random.default_rng(self.config.random_seed)
        label = model_name or self.config.model_name

        skf = StratifiedKFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.random_seed,
        )
        splits = list(skf.split(X, y))

        curve_results: dict[float, list] = {}

        for frac in fractions:
            logger.info("Learning curve: frac=%.0f%%", frac * 100)
            fold_metrics = []

            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                X_test, y_test = X[test_idx], y[test_idx]

                n_sub = max(2, int(len(train_idx) * frac))
                sub_idx = rng.choice(train_idx, size=n_sub, replace=False)
                X_train_sub, y_train_sub = X[sub_idx], y[sub_idx]

                model = model_factory(self.config)

                logger.info(
                    "  Fold %d/%d | frac=%.0f%% | n_train=%d",
                    fold_idx + 1, self.config.n_folds, frac * 100, n_sub,
                )

                with Timer() as train_timer:
                    model.fit(X_train_sub, y_train_sub)
                with Timer() as infer_timer:
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)

                if self.task_type == "classification":
                    metrics = compute_classification_metrics(
                        y_true=y_test,
                        y_pred=y_pred,
                        y_prob=y_prob,
                        training_time=train_timer.elapsed,
                        inference_time=infer_timer.elapsed,
                        n_train=n_sub,
                        n_parameters=model.n_parameters,
                    )
                else:
                    metrics = compute_regression_metrics(
                        y_true=y_test,
                        y_pred=y_pred,
                        training_time=train_timer.elapsed,
                        inference_time=infer_timer.elapsed,
                        n_train=n_sub,
                        n_parameters=model.n_parameters,
                    )

                fold_metrics.append(metrics)

            curve_results[frac] = fold_metrics
            self._save_learning_curve_checkpoint(label, frac, fold_metrics)

        return curve_results

    def _save_learning_curve_checkpoint(
        self,
        model_name: str,
        frac: float,
        fold_metrics: list,
    ) -> None:
        """Save completed fraction results immediately as a checkpoint CSV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = self.config.experiment_id
        frac_pct = int(frac * 100)
        path = LOGS_DIR / f"{exp_id}__lc_{model_name}__frac{frac_pct:03d}__{timestamp}.csv"

        rows = []
        for fold_idx, m in enumerate(fold_metrics):
            rows.append({
                "experiment_id": exp_id,
                "model_name": model_name,
                "fraction": frac,
                "fold": fold_idx,
                "timestamp": timestamp,
                **m.to_dict(),
            })

        if rows:
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            logger.info("Checkpoint saved: %s", path)

    def compare_models(
        self,
        models: dict[str, BaseModel],
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, list[ClassificationMetrics | RegressionMetrics]]:
        """Run cross-validation for multiple models on the same data/splits.

        Args:
            models: Dict of {model_name: model_instance}.
            X: Feature matrix.
            y: Labels.

        Returns:
            Dict of {model_name: list of fold metrics}.
        """
        all_results = {}

        for name, model in models.items():
            logger.info("Running %s...", name)
            fold_metrics = self.run_cross_validation(model, X, y)
            all_results[name] = fold_metrics

            _log_model_summary(name, fold_metrics, self.task_type)

        self._save_results(all_results)
        return all_results

    def _save_results(self, all_results: dict[str, list]) -> None:
        """Save experiment results to CSV and JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = self.config.experiment_id

        csv_path = LOGS_DIR / f"{exp_id}__{timestamp}.csv"
        rows = []
        for model_name, fold_metrics in all_results.items():
            for fold_idx, m in enumerate(fold_metrics):
                row = {
                    "experiment_id": exp_id,
                    "model_name": model_name,
                    "model_type": self.config.model_type,
                    "fold": fold_idx,
                    "timestamp": timestamp,
                    **m.to_dict(),
                }
                rows.append(row)

        if rows:
            fieldnames = list(rows[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            logger.info("Results saved to %s", csv_path)

        json_path = LOGS_DIR / f"{exp_id}__{timestamp}.json"
        json_data = {
            "config": {
                "use_case": self.config.use_case,
                "dataset": self.config.dataset_name,
                "n_folds": self.config.n_folds,
                "n_features": self.config.n_features,
                "random_seed": self.config.random_seed,
            },
            "models": {
                name: [m.to_dict() for m in metrics]
                for name, metrics in all_results.items()
            },
        }
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summarize_metrics(m: ClassificationMetrics | RegressionMetrics) -> str:
    if isinstance(m, ClassificationMetrics):
        return f"acc={m.accuracy:.4f} f1={m.f1_macro:.4f} auc={m.auc_roc or 'N/A'} t_train={m.training_time_seconds:.2f}s"
    return f"rmse={m.rmse:.4f} r2={m.r2:.4f} t_train={m.training_time_seconds:.2f}s"


def _log_model_summary(
    name: str,
    fold_metrics: list,
    task_type: str,
) -> None:
    if task_type == "classification":
        accs = [m.accuracy for m in fold_metrics]
        f1s = [m.f1_macro for m in fold_metrics]
        times = [m.training_time_seconds for m in fold_metrics]
        logger.info(
            "[%s] acc=%.4f±%.4f  f1=%.4f±%.4f  avg_train_time=%.2fs",
            name,
            np.mean(accs), np.std(accs),
            np.mean(f1s), np.std(f1s),
            np.mean(times),
        )
    else:
        rmses = [m.rmse for m in fold_metrics]
        r2s = [m.r2 for m in fold_metrics]
        logger.info(
            "[%s] rmse=%.4f±%.4f  r2=%.4f±%.4f",
            name,
            np.mean(rmses), np.std(rmses),
            np.mean(r2s), np.std(r2s),
        )


def aggregate_fold_metrics(
    fold_metrics: list[ClassificationMetrics | RegressionMetrics],
) -> dict[str, dict[str, float]]:
    """Aggregate metrics across folds into mean ± std.

    Returns:
        Dict of {metric_name: {"mean": ..., "std": ...}}.
    """
    all_dicts = [m.to_dict() for m in fold_metrics]
    result = {}
    for key in all_dicts[0]:
        values = [d[key] for d in all_dicts if d[key] is not None]
        if values and isinstance(values[0], (int, float)):
            result[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
    return result
