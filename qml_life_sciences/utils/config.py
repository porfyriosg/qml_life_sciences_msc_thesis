"""Global configuration and constants for the QML Life Sciences project."""

from dataclasses import dataclass, field
from pathlib import Path


# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
LOGS_DIR = RESULTS_DIR / "logs"

# Ensure output dirs exist
for d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# --- Experiment Defaults ---
RANDOM_SEED = 42
N_FOLDS = 5
LEARNING_CURVE_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]

# Quantum circuit defaults
DEFAULT_N_QUBITS = 8
DEFAULT_N_LAYERS = 4
DEFAULT_MAX_ITERATIONS = 200
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_QUANTUM_DEVICE = "default.qubit"  # or "lightning.qubit" for speed


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    use_case: str  # "drug_discovery" or "genomics"
    model_type: str  # "classical" or "quantum"
    model_name: str  # e.g. "random_forest", "vqc", "qsvm"
    dataset_name: str  # e.g. "bbbp", "tox21", "gene_expression"

    # Data
    n_features: int = DEFAULT_N_QUBITS
    test_size: float = 0.2
    random_seed: int = RANDOM_SEED

    # CV
    n_folds: int = N_FOLDS
    learning_curve_fractions: list[float] = field(
        default_factory=lambda: list(LEARNING_CURVE_FRACTIONS)
    )

    # Quantum-specific
    n_qubits: int = DEFAULT_N_QUBITS
    n_layers: int = DEFAULT_N_LAYERS
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    learning_rate: float = DEFAULT_LEARNING_RATE
    quantum_device: str = DEFAULT_QUANTUM_DEVICE

    # Hyperparameters (model-specific, stored as dict)
    hyperparams: dict = field(default_factory=dict)

    @property
    def experiment_id(self) -> str:
        return f"{self.use_case}__{self.model_name}__{self.dataset_name}"
