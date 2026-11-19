"""Loading and partitioning of the released tabular variants."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .features import CSV_COLUMNS, FEATURE_COLUMNS
from .splits import load_masks

log = logging.getLogger(__name__)

META_COLUMNS = ("imsi", "reservation_id", "cluster", "start_timestamp", "label")


def load_variant(csv_path: Path) -> pd.DataFrame:
    """Load a 77-column variant CSV (``real_strong.csv`` / ``real_stealth.csv``).

    Every evaluation script loads the variants through this function (pandas'
    default parser, float64 features).
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"dataset CSV not found: {csv_path}")
    log.info("Loading %s", csv_path)
    df = pd.read_csv(csv_path)
    if list(df.columns) != CSV_COLUMNS:
        raise ValueError(f"{csv_path.name}: unexpected columns; expected the 77-column "
                         "schema f0_mean..f24_p95, attack_type, label")
    if df[FEATURE_COLUMNS].isna().any().any():
        raise ValueError(f"{csv_path.name}: feature columns contain NaN")
    log.info("  %s rows x %d columns", f"{len(df):,}", df.shape[1])
    return df


def load_meta(meta_path: Path, n_rows: int | None = None) -> pd.DataFrame:
    """Load ``windows_meta.csv`` and check it is row-aligned with ``n_rows`` windows."""
    meta_path = Path(meta_path)
    if not meta_path.is_file():
        raise FileNotFoundError(f"windows_meta.csv not found: {meta_path}")
    meta = pd.read_csv(meta_path, dtype={"imsi": str, "reservation_id": str})
    missing = set(META_COLUMNS) - set(meta.columns)
    if missing:
        raise ValueError(f"{meta_path.name} lacks columns {sorted(missing)}")
    if n_rows is not None and len(meta) != n_rows:
        raise ValueError(f"meta has {len(meta):,} rows but the data has {n_rows:,}")
    return meta


@dataclass
class Split:
    """Scaled train/test arrays with binary and multi-class targets."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train_bin: np.ndarray
    y_test_bin: np.ndarray
    y_train_multi: np.ndarray
    y_test_multi: np.ndarray
    classes: np.ndarray
    feature_names: list[str]


def scale_partition(X: np.ndarray, train: np.ndarray,
                    test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a StandardScaler on ``X[train]`` and transform both partitions.

    The single scaling path of every tabular experiment (float64 arrays).
    """
    scaler = StandardScaler()
    return scaler.fit_transform(X[train]), scaler.transform(X[test])


def make_split(df: pd.DataFrame, mask_dir: Path,
               columns: list[str] | None = None) -> Split:
    """Partition ``df`` with the released masks; fit StandardScaler on train only.

    ``columns`` restricts the feature columns (used by the ablation study).
    Multi-class labels are encoded with ``LabelEncoder`` fitted on all four
    class names, giving the alphabetical order benign, dos, phy, scan.
    """
    columns = list(FEATURE_COLUMNS if columns is None else columns)
    train, test = load_masks(mask_dir, n_rows=len(df))
    X_train, X_test = scale_partition(df[columns].to_numpy(dtype=np.float64), train, test)
    le = LabelEncoder().fit(df["attack_type"])
    y_multi = le.transform(df["attack_type"])
    y_bin = df["label"].to_numpy()
    log.info("Train %s | test %s | test attack prevalence %.4f",
             f"{train.sum():,}", f"{test.sum():,}", y_bin[test].mean())
    return Split(X_train, X_test, y_bin[train], y_bin[test],
                 y_multi[train], y_multi[test], le.classes_, columns)
