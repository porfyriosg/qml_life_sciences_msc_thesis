# Quantum Machine Learning in the Life Sciences

MSc Thesis — *Quantum Computing and Quantum Machine Learning in the Life Sciences*  
**Author:** Porfyrios Giannakouris  
MSc Quantum Computing & Quantum Technologies, Democritus University of Thrace  

---

## Overview

Rigorous benchmarking of classical ML versus quantum ML across two life sciences use cases:

| Use Case | Dataset | Task | Samples | Features |
|---|---|---|---|---|
| Drug Discovery | BBBP (MoleculeNet) | Binary classification (BBB permeability) | 2,039 | 8 PCA from 1024-bit Morgan fingerprints |
| Genomics | UCI Gene Expression Cancer RNA-Seq | 5-class cancer type classification | 801 | 8 PCA from 20,531 genes |

**Models compared:**

| Type | Models |
|---|---|
| Classical | Random Forest, XGBoost, SVM (RBF), MLP (genomics only) |
| Quantum | VQC (Variational Quantum Classifier), QSVM (Quantum Kernel SVM) |

All quantum experiments run on PennyLane's `default.qubit` statevector simulator (8 qubits, 4 layers). No quantum hardware account or API key required.

---

## Repository Structure

```
qml_life_sciences_thesis/
├── qml_life_sciences/          # Main library
│   ├── drug_discovery/         # Data loading, features, classical & quantum models
│   ├── genomics/               # Data loading, features, classical & quantum models
│   ├── molecular_modeling/     # VQE demo (optional)
│   └── utils/                  # Benchmarking harness, metrics, visualization, config
├── experiments/
│   ├── drug_discovery/
│   │   ├── run_classical.py    # RF, XGBoost, SVM baselines + learning curves
│   │   └── run_quantum.py      # VQC, QSVM (200-sample subsample) + learning curves
│   ├── genomics/
│   │   ├── run_classical.py    # RF, XGBoost, SVM, MLP baselines + learning curves
│   │   └── run_quantum.py      # VQC, QSVM (200-sample subsample) + learning curves
│   ├── vqe_demo/
│   │   └── run_vqe.py          # Optional VQE energy demo (H₂, LiH)
│   └── analyze_results.py      # Statistical analysis + figure/table generation
├── results/
│   ├── logs/                   # CSV/JSON experiment outputs (auto-generated)
│   ├── figures/                # PDF + PNG thesis figures (auto-generated)
│   └── tables/                 # LaTeX tables + stats CSV (auto-generated)
├── data/                       # Downloaded datasets (auto-generated, not committed)
├── tests/
│   └── test_benchmarking.py
├── requirements.txt
└── requirements-quantum.txt
```

---

## Setup

### Requirements

- Python 3.10+

### Install

```bash
# Clone the repo
git clone <repo-url>
cd qml_life_sciences_thesis

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the package and all dependencies
pip install -e ".[quantum,jupyter]"
```

This installs both `requirements.txt` (classical stack) and `requirements-quantum.txt` (PennyLane).  
Key versions: Python 3.13, PennyLane 0.45, scikit-learn 1.9, RDKit 2026.03, numpy 2.x.

---

## How to Run

### 1. Classical baselines

Datasets are downloaded automatically on first run.

```bash
# Drug discovery (BBBP) — RF, XGBoost, SVM + learning curves (~2 min)
python experiments/drug_discovery/run_classical.py

# Genomics (RNA-Seq) — RF, XGBoost, SVM, MLP + learning curves (~3 min)
python experiments/genomics/run_classical.py
```

Results saved to `results/logs/` as CSV and JSON.

### 2. Quantum models

Uses a stratified 200-sample subsample to keep runtime manageable.  
**Warning:** each script takes 2–5 hours on a laptop CPU.

```bash
# Drug discovery — VQC then QSVM + learning curves
python experiments/drug_discovery/run_quantum.py

# Genomics — VQC then QSVM + learning curves
python experiments/genomics/run_quantum.py
```

Results are checkpointed to `results/logs/` after each training-size fraction — a crash only loses the current fraction, not all prior work.

### 3. Statistical analysis & figures

Requires all four experiment scripts above to have completed.

```bash
python experiments/analyze_results.py
```

Generates in `results/figures/<use_case>/`:
- Accuracy, F1 (macro), AUC-ROC comparison bar charts (classical vs quantum)
- Training time comparison (log scale)
- Learning curves — real curves for both classical and quantum
- Multi-metric radar chart

Generates in `results/tables/`:
- `drug_discovery_comparison_table.tex` — ready to paste into LaTeX
- `genomics_comparison_table.tex`
- `statistical_tests.csv` — paired Wilcoxon signed-rank (exact, n=5) p-values across all metrics

### 4. Tests

```bash
pytest tests/ -v
```

---

## Evaluation Protocol

| Setting | Value |
|---|---|
| Cross-validation | 5-fold stratified |
| Metrics | Accuracy, F1 (macro), AUC-ROC, training time, inference time |
| Learning curve fractions | 10%, 25%, 50%, 75%, 100% |
| Statistical test | Paired Wilcoxon signed-rank, exact method |
| Quantum feature scaling | MinMaxScaler → [0, π] for angle encoding |
| Quantum subsample | 200 stratified samples (QSVM kernel is O(n²)) |
| Quantum device | `default.qubit` statevector simulator (no hardware) |
| Qubits / layers | 8 qubits, 4 StronglyEntanglingLayers |

---

## Key Findings (Summary)

- **QSVM matches the best classical model** (XGBoost) on BBBP drug discovery — accuracy 0.875 vs 0.872, Wilcoxon p=1.0 (no significant difference)
- **VQC underperforms** on both datasets due to barren plateau effects at 8 qubits / 4 layers, a known NISQ-era limitation documented in the thesis
- **Quantum training time** is 3–4 orders of magnitude higher than classical on a CPU simulator
- **QSVM on genomics** achieves F1=0.876, AUC-ROC=0.993 vs XGBoost's 0.982/0.9996 — competitive but not superior
- No comparisons reach p<0.05; the minimum exact p-value with n=5 folds is 0.0625

All quantum experiments use simulators. Results reflect algorithmic differences, not hardware noise.

---

## Citation

```
Giannakouris, P. (2026). Quantum Computing and Quantum Machine Learning
in the Life Sciences. MSc Thesis, Democritus University of Thrace.
```
