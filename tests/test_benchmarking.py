"""Tests for the benchmarking framework using synthetic data.

Run with: pytest tests/test_benchmarking.py -v
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from qml_life_sciences.utils.config import ExperimentConfig
from qml_life_sciences.utils.benchmarking import BaseModel, BenchmarkRunner
from qml_life_sciences.utils.metrics import (
    compute_classification_metrics,
    Timer,
)


# --- Dummy model for testing the harness ---

class DummyClassifier(BaseModel):
    """Always predicts the majority class. Used to test the framework."""

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self._majority_class = None

    @property
    def name(self) -> str:
        return "DummyClassifier"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        values, counts = np.unique(y, return_counts=True)
        self._majority_class = values[np.argmax(counts)]
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self._majority_class)


# --- Tests ---

@pytest.fixture
def config():
    return ExperimentConfig(
        use_case="test",
        model_type="classical",
        model_name="dummy",
        dataset_name="synthetic",
        n_features=4,
        n_folds=3,
    )


@pytest.fixture
def synthetic_data():
    X, y = make_classification(
        n_samples=100, n_features=4, n_classes=2, random_state=42,
    )
    return X, y


def test_timer():
    with Timer() as t:
        _ = sum(range(1000))
    assert t.elapsed > 0
    assert t.elapsed < 1  # Should be near-instant


def test_compute_classification_metrics(synthetic_data):
    X, y = synthetic_data
    # Perfect predictions
    metrics = compute_classification_metrics(
        y_true=y, y_pred=y, y_prob=None,
        training_time=0.1, inference_time=0.01, n_train=80,
    )
    assert metrics.accuracy == 1.0
    assert metrics.f1_macro == 1.0
    assert metrics.training_time_seconds == 0.1


def test_benchmark_runner_cv(config, synthetic_data):
    X, y = synthetic_data
    model = DummyClassifier(config)
    runner = BenchmarkRunner(config, task_type="classification")

    fold_metrics = runner.run_cross_validation(model, X, y)

    assert len(fold_metrics) == config.n_folds
    for m in fold_metrics:
        assert 0 <= m.accuracy <= 1
        assert m.training_time_seconds >= 0
        assert m.n_test_samples > 0


def test_benchmark_runner_compare(config, synthetic_data):
    X, y = synthetic_data
    models = {
        "Dummy1": DummyClassifier(config),
        "Dummy2": DummyClassifier(config),
    }
    runner = BenchmarkRunner(config, task_type="classification")
    results = runner.compare_models(models, X, y)

    assert "Dummy1" in results
    assert "Dummy2" in results
    assert len(results["Dummy1"]) == config.n_folds


def test_learning_curve(config, synthetic_data):
    X, y = synthetic_data
    config.learning_curve_fractions = [0.25, 0.5, 1.0]

    runner = BenchmarkRunner(config, task_type="classification")
    curve = runner.run_learning_curve(DummyClassifier, X, y)

    assert set(curve.keys()) == {0.25, 0.5, 1.0}
    for frac, metrics_list in curve.items():
        assert len(metrics_list) == config.n_folds
