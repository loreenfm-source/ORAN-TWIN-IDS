"""Statistical feature extraction (Stage 4).

Every ``(T, F)`` window becomes 3*F values: per signal the mean, the
population standard deviation (``ddof=0``) and the 95th percentile (linear
interpolation). The released CSVs were produced with a per-window loop under
numpy 1.26: means and standard deviations in float32 (the window dtype) and
percentiles in float64. The vectorised implementation below reproduces those
values bit for bit, independently of the installed numpy version (numpy >= 2
returns float32 percentiles for float32 input, which would change the CSVs in
the last digits).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, N_FEATURES

DEFAULT_CHUNK = 100_000
PERCENTILE = 95


def percentile_linear(x: np.ndarray, q: float = PERCENTILE) -> np.ndarray:
    """``np.percentile(x, q, axis=-1)`` with numpy 1.26 float32 semantics.

    The neighbouring order statistics are subtracted in float32 and the
    interpolation is carried out in float64, following ``numpy.lib.
    function_base._lerp`` in numpy 1.26.
    """
    n = x.shape[-1]
    quantile = np.float64(q) / 100
    # numpy's virtual index for the "linear" method (alpha = beta = 1).
    virtual = n * quantile + (1 + quantile * (1 - 1 - 1)) - 1
    lo = int(np.floor(virtual))
    hi = min(lo + 1, n - 1)
    gamma = virtual - lo
    s = np.sort(x, axis=-1)
    a, b = s[..., lo], s[..., hi]
    diff = (b - a).astype(np.float64)
    if gamma >= 0.5:
        return b.astype(np.float64) - diff * (1 - gamma)
    return a.astype(np.float64) + diff * gamma


def window_statistics(windows: np.ndarray) -> pd.DataFrame:
    """Return the 75 statistical columns for a ``(n, T, F)`` block of windows."""
    if windows.ndim != 3 or windows.shape[2] != N_FEATURES:
        raise ValueError(f"expected (n, T, {N_FEATURES}) windows, got {windows.shape}")
    # (n, F, T) contiguous, so every reduction runs over a contiguous time axis,
    # exactly as it does for a single column ``window[:, f]``.
    x = np.ascontiguousarray(np.asarray(windows, dtype=np.float32).transpose(0, 2, 1))
    stats = {
        "mean": x.mean(axis=2),
        "std": x.std(axis=2),
        "p95": percentile_linear(x),
    }
    data = {c: stats[c.split("_", 1)[1]][:, int(c.split("_", 1)[0][1:])]
            for c in FEATURE_COLUMNS}
    return pd.DataFrame(data, columns=FEATURE_COLUMNS)


def write_feature_csv(windows: np.ndarray, attack_types: np.ndarray, path: Path,
                      chunk: int = DEFAULT_CHUNK, progress: bool = False) -> None:
    """Write the 77-column CSV (features, ``attack_type``, ``label``) in chunks."""
    n = windows.shape[0]
    if len(attack_types) != n:
        raise ValueError("attack_types and windows have different lengths")
    path = Path(path)
    starts = range(0, n, chunk)
    if progress:
        from tqdm import tqdm
        starts = tqdm(starts, desc="Extracting features", total=-(-n // chunk))
    for s in starts:
        e = min(s + chunk, n)
        df = window_statistics(windows[s:e])
        types = np.asarray(attack_types[s:e], dtype=object)
        df["attack_type"] = types
        df["label"] = (types != "benign").astype(np.int64)
        df.to_csv(path, mode="w" if s == 0 else "a", header=(s == 0), index=False)
