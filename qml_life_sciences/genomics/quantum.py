"""Quantum ML models for genomics benchmarking.

The VQC and QSVM implementations are task-agnostic, so we re-export
them from drug_discovery.quantum. This module can hold genomics-specific
quantum architectures if needed later.
"""

from ..drug_discovery.quantum import VQCModel, QSVMModel  # noqa: F401
