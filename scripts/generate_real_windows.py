#!/usr/bin/env python3
"""Stage 2 - sliding windows over the point-wise table.

Rows are grouped by (reservation_id, imsi) - the source dataset reuses the
same eight IMSIs in every reservation - and sorted by timestamp within each
group. Windows of ``--window`` rows are taken every ``--stride`` rows; groups
shorter than one window are skipped. Groups are visited in sorted key order
(reservation identifier, then IMSI), which defines the row order of every
released file.

A ``StandardScaler`` is fitted on all windows flattened to ``(N*T, F)`` and
applied to every window.

Outputs (in ``--output_dir``):

* ``windows_scaled.npz`` - key ``windows``, float32, shape (N, T, 25)
* ``windows_meta.csv``   - imsi, reservation_id, cluster, start_timestamp,
  label; one row per window, row-aligned with the NPZ and the CSV variants
* ``scaler.joblib`` and ``feature_names.json``
* ``windows_raw.npz`` (unscaled) only with ``--save_raw``

Resources: the point-wise features are held in memory as float32 (about 9 GB
for the 90.4 M source rows, plus about 2 GB of keys); windows are assembled in
a disk-backed memmap that needs about 18 GB of free space in ``--output_dir``
(about 36 GB with ``--save_raw``). Fitting the scaler reads all windows.

Example::

    python scripts/generate_real_windows.py \
        --pointwise_csv outputs/pointwise/pointwise_dataset.csv \
        --output_dir outputs/windows
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from oran_twin_ids.features import RAW_FEATURES  # noqa: E402

log = logging.getLogger("windows")

WINDOW = 50
STRIDE = 25
EXPECTED_WINDOWS_FULL = 3_606_121  # full source dataset
CSV_CHUNK_ROWS = 5_000_000
SCALE_CHUNK = 200_000
KEY_COLUMNS = ["reservation_id", "imsi", "timestamp", "label"]


class PointwiseKeys:
    """Compact per-row keys of the point-wise table.

    Reservation identifiers are stored as integer codes (``names[code]``) and
    timestamps as ``datetime64[ns]``, which keeps memory small for 90 M rows.
    """

    def __init__(self, res_code: np.ndarray, names: list[str], imsi: np.ndarray,
                 ts: np.ndarray, label: np.ndarray):
        self.res_code, self.names, self.imsi, self.ts, self.label = res_code, names, imsi, ts, label

    def __len__(self) -> int:
        return len(self.res_code)


def read_pointwise(path: Path) -> tuple[np.ndarray, PointwiseKeys]:
    """Return ``(features float32 (n, 25), keys)`` from the point-wise CSV."""
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in RAW_FEATURES + KEY_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"{path} lacks required columns: {missing}")
    feats, codes, imsis, stamps, labels = [], [], [], [], []
    code_of: dict[str, int] = {}
    reader = pd.read_csv(path, usecols=RAW_FEATURES + KEY_COLUMNS, chunksize=CSV_CHUNK_ROWS,
                         dtype={"reservation_id": str, "timestamp": str})
    for chunk in tqdm(reader, desc="Reading point-wise CSV", unit="chunk"):
        feats.append(chunk[RAW_FEATURES].to_numpy(np.float32))
        res = chunk["reservation_id"].to_numpy()
        for r in pd.unique(res):
            code_of.setdefault(r, len(code_of))
        codes.append(pd.Series(res).map(code_of).to_numpy(np.int32))
        imsis.append(chunk["imsi"].to_numpy(np.int64))
        stamps.append(pd.to_datetime(chunk["timestamp"], format="ISO8601").to_numpy())
        labels.append(chunk["label"].to_numpy(np.int64))
    keys = PointwiseKeys(np.concatenate(codes), list(code_of), np.concatenate(imsis),
                         np.concatenate(stamps), np.concatenate(labels))
    return np.concatenate(feats), keys


def window_starts(n_rows: int, window: int, stride: int) -> range:
    """Start offsets of all complete windows in a series of ``n_rows`` rows."""
    return range(0, n_rows - window + 1, stride) if n_rows >= window else range(0)


def ordered_groups(keys: PointwiseKeys) -> list[np.ndarray]:
    """Row indices of each (reservation_id, imsi) group, sorted by timestamp.

    Groups are ordered by reservation identifier (string order), then IMSI.
    """
    rank = np.empty(len(keys.names), dtype=np.int64)
    rank[np.argsort(np.array(keys.names, dtype=object), kind="stable")] = np.arange(len(keys.names))
    frame = pd.DataFrame({"res": rank[keys.res_code], "imsi": keys.imsi})
    groups = []
    for _, idx in sorted(frame.groupby(["res", "imsi"], sort=True).indices.items()):
        groups.append(idx[np.argsort(keys.ts[idx], kind="stable")])
    return groups


def build_windows(features: np.ndarray, keys: PointwiseKeys, cluster_map: dict,
                  out_path: Path, window: int, stride: int) -> tuple[np.memmap, pd.DataFrame]:
    """Write raw windows to a memmap at ``out_path`` and return it with the metadata."""
    groups = ordered_groups(keys)
    n_windows = sum(len(window_starts(len(g), window, stride)) for g in groups)
    if n_windows == 0:
        raise ValueError(f"no group has at least {window} rows")
    log.info("%d (reservation, IMSI) groups -> %s windows", len(groups), f"{n_windows:,}")
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32,
                                    shape=(n_windows, window, features.shape[1]))
    first = np.empty(n_windows, dtype=np.int64)  # point-wise row of each window start
    k = 0
    for idx in tqdm(groups, desc="Building windows"):
        arr = features[idx]
        for s in window_starts(len(idx), window, stride):
            out[k] = arr[s:s + window]
            first[k] = idx[s]
            k += 1
    names = np.array(keys.names, dtype=object)[keys.res_code[first]]
    meta = pd.DataFrame({
        "imsi": keys.imsi[first],
        "reservation_id": names,
        "cluster": pd.Series(names).map(cluster_map).to_numpy(),
        # Stage 1 wrote Timestamp.isoformat(); re-formatting reproduces those strings.
        "start_timestamp": [t.isoformat() for t in pd.DatetimeIndex(keys.ts[first])],
        "label": keys.label[first],
    })
    unmapped = meta.loc[meta["cluster"].isna(), "reservation_id"].unique()
    if len(unmapped):
        raise ValueError(f"{len(unmapped)} reservations missing from the cluster map, "
                         f"e.g. {list(unmapped[:5])}")
    return out, meta


def scale_in_place(windows: np.memmap, chunk: int = SCALE_CHUNK) -> StandardScaler:
    """Fit a StandardScaler on all flattened windows and scale them in place."""
    n, _, f = windows.shape
    scaler = StandardScaler().fit(windows.reshape(-1, f))
    for s in tqdm(range(0, n, chunk), desc="Scaling"):
        e = min(s + chunk, n)
        # In-place float32 arithmetic, as StandardScaler.transform did in the
        # scikit-learn version (1.4.2) that produced the released archive;
        # newer versions round differently in the last bit.
        x = np.array(windows[s:e], dtype=np.float32)
        x -= scaler.mean_
        x /= scaler.scale_
        windows[s:e] = x
    windows.flush()
    return scaler


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Stage 2: sliding windows and scaling.")
    p.add_argument("--pointwise_csv", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, default=Path("outputs/windows"))
    p.add_argument("--cluster_map", type=Path,
                   default=REPO_ROOT / "metadata" / "reservation_cluster_map.json")
    p.add_argument("--window", type=int, default=WINDOW, help="window length T in rows")
    p.add_argument("--stride", type=int, default=STRIDE, help="stride in rows")
    p.add_argument("--save_raw", action="store_true", help="also write unscaled windows_raw.npz")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for path in (args.pointwise_csv, args.cluster_map):
        if not path.is_file():
            p.error(f"file not found: {path}")
    if args.window < 1 or args.stride < 1:
        p.error("--window and --stride must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cluster_map = json.loads(args.cluster_map.read_text(encoding="utf-8"))
    features, keys = read_pointwise(args.pointwise_csv)
    log.info("%s point-wise rows", f"{len(keys):,}")

    tmp = args.output_dir / "_windows_tmp.npy"
    windows, meta = build_windows(features, keys, cluster_map, tmp, args.window, args.stride)
    del features, keys
    if len(meta) != EXPECTED_WINDOWS_FULL:
        log.warning("%s windows (the full source dataset gives %s)",
                    f"{len(meta):,}", f"{EXPECTED_WINDOWS_FULL:,}")
    if args.save_raw:
        np.savez_compressed(args.output_dir / "windows_raw.npz", windows=windows)

    scaler = scale_in_place(windows)
    np.savez_compressed(args.output_dir / "windows_scaled.npz", windows=windows)
    meta.to_csv(args.output_dir / "windows_meta.csv", index=False)
    joblib.dump(scaler, args.output_dir / "scaler.joblib")
    (args.output_dir / "feature_names.json").write_text(
        json.dumps({"raw": RAW_FEATURES}, indent=2), encoding="utf-8")
    del windows
    tmp.unlink()
    log.info("windows_scaled.npz: %s | clusters: %s", (len(meta), args.window, len(RAW_FEATURES)),
             meta["cluster"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
