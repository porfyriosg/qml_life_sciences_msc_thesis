"""Run classical baselines for drug discovery use case.

Usage:
    python experiments/drug_discovery/run_classical.py
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from qml_life_sciences.utils.config import ExperimentConfig
from qml_life_sciences.utils.benchmarking import BenchmarkRunner
from qml_life_sciences.drug_discovery.classical import (
    RandomForestModel,
    XGBoostModel,
    SVMModel,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    # --- Configuration ---
    config = ExperimentConfig(
        use_case="drug_discovery",
        model_type="classical",
        model_name="all_classical",
        dataset_name="bbbp",  # Change to "esol" for regression
        n_features=8,
        n_qubits=8,
    )

    # --- Load Data ---
    from qml_life_sciences.drug_discovery.data import load_bbbp
    dataset = load_bbbp(n_features=config.n_features)
    X, y = dataset.X, dataset.y
    logger.info("Loaded %s: %d samples, %d features", dataset.dataset_name, len(y), X.shape[1])

    # --- Define Models ---
    models = {
        "Random Forest": RandomForestModel(config),
        "XGBoost": XGBoostModel(config),
        "SVM (RBF)": SVMModel(config),
    }

    # --- Run Benchmark ---
    runner = BenchmarkRunner(config, task_type="classification")

    logger.info("=" * 60)
    logger.info("Drug Discovery - Classical Baselines")
    logger.info("Dataset: %s | Features: %d | Folds: %d", config.dataset_name, config.n_features, config.n_folds)
    logger.info("=" * 60)

    results = runner.compare_models(models, X, y)

    logger.info("Running learning curves (checkpointed per fraction)...")
    for name, model_cls in [
        ("Random Forest", RandomForestModel),
        ("XGBoost", XGBoostModel),
        ("SVM (RBF)", SVMModel),
    ]:
        logger.info("Learning curve: %s", name)
        runner.run_learning_curve(model_cls, X, y, model_name=name)

    logger.info("Done! Check results/logs/ for output files.")


if __name__ == "__main__":
    main()
