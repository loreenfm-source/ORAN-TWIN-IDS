"""Reservation-level train/test partition.

Within each stratum - by default each (cluster, slicing, scheduling)
configuration cell - the first ``train_frac`` of reservations in identifier
order go to training and the rest to test, so that no reservation (and
therefore no pair of overlapping windows) contributes to both partitions and
every configuration is represented on both sides.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_TRAIN_FRAC = 0.7
DEFAULT_STRATA = ("cluster", "slicing", "scheduling")
TRAIN_MASK_FILE = "split_train_mask.npy"
TEST_MASK_FILE = "split_test_mask.npy"


def reservation_split(meta: pd.DataFrame, train_frac: float = DEFAULT_TRAIN_FRAC,
                      strata: tuple[str, ...] = DEFAULT_STRATA) -> np.ndarray:
    """Return the boolean training mask for the rows of ``meta``.

    ``meta`` must have ``reservation_id`` and the ``strata`` columns and be
    row-aligned with the windows. Reservations are ordered by identifier
    (lexicographic, which equals numeric order for ``RESERVATION-<6 digits>``).
    ``strata=("cluster",)`` gives the earlier per-cluster split.
    """
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be in (0, 1)")
    strata = list(strata)
    missing = {"reservation_id", *strata} - set(meta.columns)
    if missing:
        raise ValueError(f"meta lacks columns {sorted(missing)}")
    if meta[strata].isna().any().any():
        raise ValueError(f"meta has rows without a value in {strata}")

    res = meta["reservation_id"].astype(str).to_numpy()
    train = np.zeros(len(meta), dtype=bool)
    for _, group in meta.groupby(strata, sort=True):
        ids = sorted(group["reservation_id"].astype(str).unique())
        k = int(round(train_frac * len(ids)))
        train |= np.isin(res, ids[:k])
    check_disjoint(res, train)
    return train


def check_disjoint(reservation_ids: np.ndarray, train_mask: np.ndarray) -> None:
    """Raise if any reservation has windows on both sides of the split."""
    both = set(reservation_ids[train_mask]) & set(reservation_ids[~train_mask])
    if both:
        raise AssertionError(f"{len(both)} reservations appear in both partitions")


def load_masks(mask_dir: Path, n_rows: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load ``split_train_mask.npy`` / ``split_test_mask.npy`` and validate them."""
    mask_dir = Path(mask_dir)
    paths = [mask_dir / TRAIN_MASK_FILE, mask_dir / TEST_MASK_FILE]
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"split mask not found: {p}")
    train, test = (np.load(p, allow_pickle=False) for p in paths)
    if train.dtype != bool or test.dtype != bool or train.shape != test.shape:
        raise ValueError("split masks must be boolean arrays of equal length")
    if (train & test).any():
        raise ValueError("train and test masks overlap")
    if not (train | test).all():
        log.warning("%d rows are in neither partition", int((~(train | test)).sum()))
    if n_rows is not None and len(train) != n_rows:
        raise ValueError(f"masks have {len(train):,} rows but the data has {n_rows:,}")
    return train, test
