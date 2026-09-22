"""Quantum ML models for drug discovery benchmarking.

Implements:
1. Variational Quantum Classifier (VQC) with angle encoding
2. Quantum Kernel SVM (QSVM) using quantum kernel estimation

Both use PennyLane as the quantum framework.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ..utils.benchmarking import BaseModel
from ..utils.config import ExperimentConfig

logger = logging.getLogger(__name__)

# Lazy import PennyLane (only when quantum models are used)
_pennylane_imported = False
qml = None
qnp = None


def _ensure_pennylane():
    global _pennylane_imported, qml, qnp
    if not _pennylane_imported:
        import pennylane as _qml
        import pennylane.numpy as _qnp
        qml = _qml
        qnp = _qnp
        _pennylane_imported = True


# ---------------------------------------------------------------------------
# Variational Quantum Classifier (VQC)
# ---------------------------------------------------------------------------

class VQCModel(BaseModel):
    """Variational Quantum Classifier using PennyLane.

    Architecture:
    - Angle encoding: each feature maps to a rotation gate
    - Strongly entangling layers as the variational ansatz
    - Measurement: expectation values of Pauli-Z on each qubit
    - Classical post-processing: softmax over measurements
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        _ensure_pennylane()

        self.n_qubits = config.n_qubits
        self.n_layers = config.n_layers
        self.max_iter = config.max_iterations
        self.lr = config.learning_rate
        self.weights = None
        self.bias = None
        self._classes = None

        self.dev = qml.device(config.quantum_device, wires=self.n_qubits)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    @property
    def name(self) -> str:
        return f"VQC (q={self.n_qubits}, L={self.n_layers})"

    @property
    def n_parameters(self) -> int | None:
        if self.weights is not None:
            return self.weights.size + (self.bias.size if self.bias is not None else 0)
        # Theoretical count: StronglyEntanglingLayers has 3 params per qubit per layer
        return self.n_layers * self.n_qubits * 3

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train VQC using gradient descent."""
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        weight_shape = qml.StronglyEntanglingLayers.shape(
            n_layers=self.n_layers, n_wires=self.n_qubits
        )
        self.weights = qnp.random.uniform(
            -np.pi, np.pi, size=weight_shape, requires_grad=True
        )
        self.bias = qnp.zeros(n_classes, requires_grad=True)

        opt = qml.GradientDescentOptimizer(stepsize=self.lr)

        y_onehot = np.eye(n_classes)[
            np.searchsorted(self._classes, y)
        ]

        for step in range(self.max_iter):
            (self.weights, self.bias), cost = opt.step_and_cost(
                lambda w, b: self._cost_fn(w, b, X, y_onehot),
                self.weights,
                self.bias,
            )
            if step % 50 == 0:
                logger.debug("VQC step %d/%d, cost=%.4f", step, self.max_iter, cost)

        self.is_fitted = True

    def _cost_fn(
        self, weights: np.ndarray, bias: np.ndarray,
        X: np.ndarray, y_onehot: np.ndarray,
    ) -> float:
        """Cross-entropy cost function."""
        predictions = self._raw_predict(X, weights, bias)
        # Clip for numerical stability
        predictions = qnp.clip(predictions, 1e-7, 1 - 1e-7)
        loss = -qnp.mean(qnp.sum(y_onehot * qnp.log(predictions), axis=1))
        return loss

    def _raw_predict(
        self, X: np.ndarray, weights: np.ndarray, bias: np.ndarray,
    ) -> np.ndarray:
        """Get raw softmax predictions."""
        logits = []
        for x in X:
            circuit_out = self.circuit(x, weights)
            # first n_classes qubits + bias; remaining qubits are discarded
            # qnp.array (not np.array) keeps this inside autograd's tracing
            n_classes = len(bias)
            raw = qnp.array(circuit_out[:n_classes]) + bias
            logits.append(raw)

        logits = qnp.array(logits)
        exp_logits = qnp.exp(logits - qnp.max(logits, axis=1, keepdims=True))
        return exp_logits / qnp.sum(exp_logits, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self._raw_predict(X, self.weights, self.bias)
        indices = np.argmax(proba, axis=1)
        return self._classes[indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        return np.array(self._raw_predict(X, self.weights, self.bias))


# ---------------------------------------------------------------------------
# Quantum Kernel SVM (QSVM)
# ---------------------------------------------------------------------------

class QSVMModel(BaseModel):
    """Quantum Support Vector Machine using quantum kernel estimation.

    Uses a quantum circuit to compute kernel values between data points,
    then feeds the kernel matrix to a classical SVM.

    The quantum kernel measures similarity in the Hilbert space of the
    quantum feature map, which can capture complex patterns that classical
    kernels miss.
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        _ensure_pennylane()

        self.n_qubits = config.n_qubits
        self.n_layers = config.hyperparams.get("kernel_layers", 2)
        self.C = config.hyperparams.get("C", 1.0)

        self.dev = qml.device(config.quantum_device, wires=self.n_qubits)
        self.svm = None
        self.X_train = None

        @qml.qnode(self.dev, interface="autograd")
        def kernel_circuit(x1, x2):
            self._feature_map(x1)
            qml.adjoint(self._feature_map)(x2)  # overlap with x2's feature map
            return qml.probs(wires=range(self.n_qubits))  # kernel = P(|0...0>)

        self.kernel_circuit = kernel_circuit

    def _feature_map(self, x: np.ndarray) -> None:
        """ZZFeatureMap-style encoding: H+RZ per qubit then nearest-neighbour ZZ."""
        for _ in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
                qml.RZ(x[i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(x[i] * x[i + 1], wires=i + 1)
                qml.CNOT(wires=[i, i + 1])

    def _compute_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute quantum kernel value between two data points."""
        probs = self.kernel_circuit(x1, x2)
        return float(probs[0])  # Probability of |0...0>

    def _compute_kernel_matrix(
        self, X1: np.ndarray, X2: np.ndarray
    ) -> np.ndarray:
        """Compute full kernel matrix between two sets of points."""
        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))
        total = n1 * n2
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self._compute_kernel(X1[i], X2[j])
                if (i * n2 + j + 1) % 100 == 0:
                    logger.debug(
                        "Kernel computation: %d/%d (%.0f%%)",
                        i * n2 + j + 1, total,
                        (i * n2 + j + 1) / total * 100,
                    )
        return K

    @property
    def name(self) -> str:
        return f"QSVM (q={self.n_qubits}, L={self.n_layers})"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.svm import SVC

        self.X_train = X.copy()
        logger.info("Computing training kernel matrix (%d x %d)...", len(X), len(X))
        K_train = self._compute_kernel_matrix(X, X)

        self.svm = SVC(kernel="precomputed", C=self.C, probability=True)
        self.svm.fit(K_train, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        K_test = self._compute_kernel_matrix(X, self.X_train)
        return self.svm.predict(K_test)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        K_test = self._compute_kernel_matrix(X, self.X_train)
        return self.svm.predict_proba(K_test)
