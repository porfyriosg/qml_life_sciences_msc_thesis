"""Dataset loading and preprocessing for drug discovery use case.

Supported datasets from MoleculeNet:
- BBBP: Blood-brain barrier permeability (binary classification)
- Tox21: Toxicity across 12 targets (multi-label, but we use single target)
- ESOL: Aqueous solubility (regression)
- FreeSolv: Hydration free energy (regression)

We recommend BBBP for classification and ESOL for regression as primary datasets.
"""

import logging
import urllib.request
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from ..utils.config import DATA_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)

DRUG_DATA_DIR = DATA_DIR / "drug_discovery"
DRUG_DATA_DIR.mkdir(parents=True, exist_ok=True)


class MoleculeDataset(NamedTuple):
    """Container for a molecular property dataset."""

    X: np.ndarray         # Feature matrix (n_samples, n_features)
    y: np.ndarray         # Labels or target values
    smiles: list[str]     # Original SMILES strings
    feature_names: list[str]
    dataset_name: str
    task_type: str        # "classification" or "regression"


_BBBP_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
)


def load_bbbp(n_features: int = 8) -> MoleculeDataset:
    """Load BBBP dataset with Morgan fingerprint features reduced via PCA.

    Blood-Brain Barrier Permeability: binary classification (~2,050 compounds).
    Downloads the MoleculeNet CSV on first call; subsequent calls use the
    cached file at DATA_DIR/drug_discovery/BBBP.csv.

    Args:
        n_features: Number of PCA components (= number of qubits for QML).

    Returns:
        MoleculeDataset with shape (n_valid, n_features).
    """
    csv_path = DRUG_DATA_DIR / "BBBP.csv"
    if not csv_path.exists():
        logger.info("Downloading BBBP dataset from MoleculeNet …")
        urllib.request.urlretrieve(_BBBP_URL, csv_path)
        logger.info("Saved to %s", csv_path)

    df = pd.read_csv(csv_path)
    # Columns: num, name, p_np (label), smiles
    smiles_col = "smiles"
    label_col = "p_np"

    raw_smiles = df[smiles_col].tolist()
    raw_labels = df[label_col].to_numpy(dtype=int)

    logger.info("Generating Morgan fingerprints for %d molecules …", len(raw_smiles))
    fps, valid_mask = generate_molecular_fingerprints(raw_smiles)

    smiles_valid = [s for s, ok in zip(raw_smiles, valid_mask) if ok]
    y = raw_labels[valid_mask]

    X_reduced, _ = reduce_features(fps, n_components=n_features)

    feature_names = [f"PC{i+1}" for i in range(n_features)]
    logger.info(
        "BBBP: %d valid molecules, %d positives (%.1f%%)",
        len(y),
        y.sum(),
        100 * y.mean(),
    )
    return MoleculeDataset(
        X=X_reduced,
        y=y,
        smiles=smiles_valid,
        feature_names=feature_names,
        dataset_name="BBBP",
        task_type="classification",
    )


def load_esol(n_features: int = 8) -> MoleculeDataset:
    """Load ESOL dataset for solubility regression.

    Aqueous solubility: regression.
    ~1,128 compounds.

    Args:
        n_features: Number of features after dimensionality reduction.

    Returns:
        MoleculeDataset with fingerprint features.
    """
    # TODO: Implement alongside BBBP
    raise NotImplementedError(
        "Implement ESOL loading. See CLAUDE.md for next steps."
    )


def load_synthetic_drug_discovery(
    n_samples: int = 500,
    n_features: int = 8,
) -> MoleculeDataset:
    """Generate synthetic binary classification data for testing the pipeline.

    Use this to validate the full pipeline before loading real BBBP data.
    """
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features - 2,
        n_redundant=2,
        n_classes=2,
        random_state=RANDOM_SEED,
    )

    smiles = [f"synthetic_mol_{i}" for i in range(n_samples)]
    feature_names = [f"PC{i+1}" for i in range(n_features)]

    return MoleculeDataset(
        X=X,
        y=y,
        smiles=smiles,
        feature_names=feature_names,
        dataset_name="synthetic_drug_discovery",
        task_type="classification",
    )


def generate_molecular_fingerprints(
    smiles_list: list[str],
    fp_type: str = "morgan",
    n_bits: int = 1024,
    radius: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate molecular fingerprints from SMILES strings via RDKit.

    Molecules that fail RDKit parsing are silently dropped; the boolean
    valid_mask can be used to align other arrays (e.g., labels) to the
    returned fingerprint matrix.

    Args:
        smiles_list: List of SMILES strings.
        fp_type: "morgan" (circular, 1024-bit) or "maccs" (166-bit).
        n_bits: Fingerprint bit length; only used for Morgan.
        radius: Morgan fingerprint radius (2 ≈ ECFP4).

    Returns:
        Tuple of:
          - fps: float32 array of shape (n_valid, n_bits)
          - valid_mask: bool array of shape (n_molecules,)
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, MACCSkeys
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")  # suppress parse warnings

    fps_list: list[np.ndarray] = []
    valid_mask = np.zeros(len(smiles_list), dtype=bool)

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if fp_type == "morgan":
            bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        elif fp_type == "maccs":
            bv = MACCSkeys.GenMACCSKeys(mol)
            n_bits = 167  # MACCS always 167 bits (bit 0 unused)
        else:
            raise ValueError(f"Unknown fp_type: {fp_type!r}. Use 'morgan' or 'maccs'.")
        fps_list.append(np.frombuffer(bv.ToBitString().encode(), dtype="uint8") - ord("0"))
        valid_mask[i] = True

    if not fps_list:
        raise ValueError("No valid molecules found in smiles_list.")

    n_invalid = (~valid_mask).sum()
    if n_invalid:
        logger.warning("Dropped %d unparseable SMILES (%.1f%%)", n_invalid, 100 * n_invalid / len(smiles_list))

    return np.array(fps_list, dtype=np.float32), valid_mask


def reduce_features(
    X: np.ndarray,
    n_components: int = 8,
    method: str = "pca",
) -> tuple[np.ndarray, object]:
    """Reduce feature dimensionality for quantum circuit compatibility.

    Args:
        X: High-dimensional feature matrix.
        n_components: Target dimensionality (= number of qubits).
        method: "pca" or "umap".

    Returns:
        Tuple of (reduced features, fitted reducer for later transform).
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=RANDOM_SEED)
    else:
        raise ValueError(f"Unknown method: {method}")

    X_reduced = reducer.fit_transform(X_scaled)
    logger.info(
        "Reduced features: %d -> %d (explained variance: %.2f%%)",
        X.shape[1],
        n_components,
        sum(reducer.explained_variance_ratio_) * 100 if hasattr(reducer, "explained_variance_ratio_") else 0,
    )
    return X_reduced, reducer
