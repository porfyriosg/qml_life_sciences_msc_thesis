"""Run quantum models for genomics use case.

Usage:
    python experiments/genomics/run_quantum.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from qml_life_sciences.utils.config import ExperimentConfig
from qml_life_sciences.utils.benchmarking import BenchmarkRunner
from qml_life_sciences.genomics.quantum import VQCModel, QSVMModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    config = ExperimentConfig(
        use_case="genomics",
        model_type="quantum",
        model_name="all_quantum",
        dataset_name="gene_expression",
        n_features=8,
        n_qubits=8,
        n_layers=4,
        max_iterations=150,
        learning_rate=0.01,
        quantum_device="default.qubit",
    )

    import numpy as np
    from qml_life_sciences.genomics.data import load_gene_expression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    dataset = load_gene_expression(n_features=config.n_features)

    n_samples = 200
    X, _, y, _ = train_test_split(
        dataset.X, dataset.y,
        train_size=n_samples,
        stratify=dataset.y,
        random_state=config.random_seed,
    )
    logger.info(
        "Subsampled %s: %d/%d samples (stratified)",
        dataset.dataset_name, len(y), len(dataset.y),
    )

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X = scaler.fit_transform(X)

    models = {
        "VQC": VQCModel(config),
        "QSVM": QSVMModel(config),
    }

    runner = BenchmarkRunner(config, task_type="classification")

    logger.info("=" * 60)
    logger.info("Genomics - Quantum Models")
    logger.info("Qubits: %d | Layers: %d | Classes: 5", config.n_qubits, config.n_layers)
    logger.info("=" * 60)

    results = runner.compare_models(models, X, y)

    logger.info("=" * 60)
    logger.info("Running learning curves (checkpointed per fraction)...")
    logger.info("=" * 60)

    for name, model_cls in [("VQC", VQCModel), ("QSVM", QSVMModel)]:
        logger.info("Learning curve: %s", name)
        runner.run_learning_curve(model_cls, X, y, model_name=name)

    logger.info("Done! Check results/logs/ for output files.")


if __name__ == "__main__":
    main()
