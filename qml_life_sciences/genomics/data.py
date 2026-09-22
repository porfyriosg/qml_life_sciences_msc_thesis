"""Dataset loading and preprocessing for genomics use case.

Primary dataset: UCI Gene Expression Cancer RNA-Seq
- 801 samples, 5 cancer types (BRCA, KIRC, COAD, LUAD, PRAD)
- 20,531 gene expression features
- Must PCA-reduce to 8-16 features for quantum circuits
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ..utils.config import DATA_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)

GENOMICS_DATA_DIR = DATA_DIR / "genomics"
GENOMICS_DATA_DIR.mkdir(parents=True, exist_ok=True)


class GenomicsDataset(NamedTuple):
    """Container for a gene expression dataset."""

    X: np.ndarray          # Feature matrix (n_samples, n_features)
    y: np.ndarray          # Encoded class labels
    class_names: list[str] # Original class names
    feature_names: list[str]
    dataset_name: str
    n_original_features: int  # Before reduction


_GENE_EXPR_URL = (
    "https://archive.ics.uci.edu/static/public/401/"
    "gene+expression+cancer+rna+seq.zip"
)


def load_gene_expression(
    n_features: int = 8,
    data_path: Path | None = None,
) -> GenomicsDataset:
    """Load and preprocess UCI Gene Expression Cancer RNA-Seq dataset.

    801 samples x 20,531 genes across 5 cancer types (BRCA, KIRC, COAD,
    LUAD, PRAD). Downloads and caches the UCI archive on first call;
    subsequent calls reuse data.csv / labels.csv in GENOMICS_DATA_DIR.

    Args:
        n_features: Number of features after PCA reduction (= n_qubits).
        data_path: Optional directory containing pre-downloaded
            data.csv / labels.csv (skips download).

    Returns:
        GenomicsDataset with shape (801, n_features).
    """
    data_dir = data_path or GENOMICS_DATA_DIR
    data_csv = data_dir / "data.csv"
    labels_csv = data_dir / "labels.csv"

    if not (data_csv.exists() and labels_csv.exists()):
        _download_gene_expression(data_dir)

    X_df = pd.read_csv(data_csv, index_col=0)
    labels_df = pd.read_csv(labels_csv, index_col=0)

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels_df["Class"])
    class_names = list(encoder.classes_)

    X_raw = X_df.to_numpy(dtype=np.float32)
    n_original_features = X_raw.shape[1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    pca = PCA(n_components=n_features, random_state=RANDOM_SEED)
    X_reduced = pca.fit_transform(X_scaled)

    logger.info(
        "Gene expression: %d -> %d features (explained variance: %.2f%%)",
        n_original_features,
        n_features,
        100 * sum(pca.explained_variance_ratio_),
    )
    logger.info(
        "Loaded %d samples across %d classes: %s",
        len(y), len(class_names), class_names,
    )

    feature_names = [f"PC{i+1}" for i in range(n_features)]

    return GenomicsDataset(
        X=X_reduced,
        y=y,
        class_names=class_names,
        feature_names=feature_names,
        dataset_name="gene_expression_cancer_rna_seq",
        n_original_features=n_original_features,
    )


def _download_gene_expression(data_dir: Path) -> None:
    """Download the UCI archive and extract data.csv / labels.csv into data_dir."""
    import shutil
    import tarfile
    import zipfile

    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "_gene_expr.zip"

    logger.info("Downloading UCI Gene Expression Cancer RNA-Seq dataset …")
    urllib.request.urlretrieve(_GENE_EXPR_URL, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        tar_name = zf.namelist()[0]  # TCGA-PANCAN-HiSeq-801x20531.tar.gz
        zf.extract(tar_name, data_dir)

    tar_path = data_dir / tar_name
    with tarfile.open(tar_path) as tf:
        tf.extractall(data_dir, filter="data")

    extracted_dir = data_dir / "TCGA-PANCAN-HiSeq-801x20531"
    shutil.move(str(extracted_dir / "data.csv"), str(data_dir / "data.csv"))
    shutil.move(str(extracted_dir / "labels.csv"), str(data_dir / "labels.csv"))

    zip_path.unlink()
    tar_path.unlink()
    shutil.rmtree(extracted_dir)

    logger.info("Saved to %s", data_dir)


def load_synthetic_genomics(
    n_samples: int = 500,
    n_features: int = 8,
    n_classes: int = 5,
) -> GenomicsDataset:
    """Generate synthetic gene expression data for testing the pipeline.

    Use this to validate the full pipeline before loading real data.
    """
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features - 2,
        n_redundant=2,
        n_classes=n_classes,
        random_state=RANDOM_SEED,
    )

    class_names = [f"Cancer_Type_{i}" for i in range(n_classes)]
    feature_names = [f"PC{i+1}" for i in range(n_features)]

    return GenomicsDataset(
        X=X,
        y=y,
        class_names=class_names,
        feature_names=feature_names,
        dataset_name="synthetic_genomics",
        n_original_features=n_features,
    )
