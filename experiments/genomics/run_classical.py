"""Run classical baselines for genomics use case.

Usage:
    python experiments/genomics/run_classical.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from qml_life_sciences.utils.config import ExperimentConfig
from qml_life_sciences.utils.benchmarking import BenchmarkRunner
from qml_life_sciences.genomics.classical import (
    RandomForestModel,
    XGBoostModel,
    SVMModel,
    MLPModel,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    config = ExperimentConfig(
        use_case="genomics",
        model_type="classical",
        model_name="all_classical",
        dataset_name="gene_expression",
        n_features=8,
        n_qubits=8,
    )

    from qml_life_sciences.genomics.data import load_gene_expression
    dataset = load_gene_expression(n_features=config.n_features)
    X, y = dataset.X, dataset.y
    logger.info("Loaded %s: %d samples, %d features", dataset.dataset_name, len(y), X.shape[1])

    models = {
        "Random Forest": RandomForestModel(config),
        "XGBoost": XGBoostModel(config),
        "SVM (RBF)": SVMModel(config),
        "MLP": MLPModel(config),
    }

    runner = BenchmarkRunner(config, task_type="classification")

    logger.info("=" * 60)
    logger.info("Genomics - Classical Baselines")
    logger.info("Dataset: %s | Features: %d | Classes: %d", config.dataset_name, config.n_features, 5)
    logger.info("=" * 60)

    results = runner.compare_models(models, X, y)

    logger.info("Running learning curves (checkpointed per fraction)...")
    for name, model_cls in [
        ("Random Forest", RandomForestModel),
        ("XGBoost", XGBoostModel),
        ("SVM (RBF)", SVMModel),
        ("MLP", MLPModel),
    ]:
        logger.info("Learning curve: %s", name)
        runner.run_learning_curve(model_cls, X, y, model_name=name)

    logger.info("Done! Check results/logs/ for output files.")


if __name__ == "__main__":
    main()
