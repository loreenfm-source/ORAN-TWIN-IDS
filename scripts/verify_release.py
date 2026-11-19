#!/usr/bin/env python3
"""Consistency checks for the released files (run after download or generation).

Checks, each reported as PASS / FAIL / INFO:

* CSV schema (77 columns), row counts and class counts (paper Table 5);
* ``windows_meta.csv`` columns, cluster assignment and per-(reservation, IMSI)
  ordering of window start times;
* split masks: SHA-256 of the released files, boolean, disjoint, complementary,
  no reservation in both, per-partition class counts of each CSV, and
  ``scripts/make_split_masks.py`` (default settings) regenerating both masks
  byte for byte from ``windows_meta.csv``;
* NPZ header: shape (N, 50, 25), float32;
* row alignment between ``windows_meta.csv`` and the NPZ: consecutive windows of
  the same (reservation, IMSI) group overlap by ``T - stride`` time steps, so
  their shared rows must be identical in the NPZ;
* row alignment between the NPZ and each CSV: statistics recomputed from
  sampled benign windows must equal the CSV rows;
* non-target leakage gate: for every ``*_p95`` column not derived from an
  injection target, the number of distinct values among attack windows relative
  to benign windows (blending artefacts inflated this ratio in v1).

Reading the NPZ streams the whole archive once (a few minutes); memory use is
dominated by loading the CSVs.

Example::

    python scripts/verify_release.py --data_dir data
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oran_twin_ids.extraction import window_statistics  # noqa: E402
from oran_twin_ids.features import (CSV_COLUMNS, FEATURE_COLUMNS, INJECTION_TARGETS,  # noqa: E402
                                    raw_index)
from oran_twin_ids.splits import TEST_MASK_FILE, TRAIN_MASK_FILE  # noqa: E402

EXPECTED_WINDOWS = 3_606_121
EXPECTED_CLASSES = {
    "real_strong": {"benign": 2_704_020, "dos": 300_226, "scan": 300_878, "phy": 300_997},
    "real_stealth": {"benign": 2_704_314, "dos": 299_960, "scan": 301_077, "phy": 300_770},
}
MASK_SHA256 = {
    TRAIN_MASK_FILE: "7ea1be46e18b59d164ba4ea492351ce10b1826fb4f456708196265750482484c",
    TEST_MASK_FILE: "e06d87c86f111ec97c67fc791f2800a05f9e38b742520632011c064fc7726b21",
}
# (train, test) windows per class under the released masks
EXPECTED_SPLIT_CLASSES = {
    "real_strong": {"benign": (1_892_710, 811_310), "dos": (209_330, 90_896),
                    "scan": (210_158, 90_720), "phy": (210_490, 90_507)},
    "real_stealth": {"benign": (1_892_077, 812_237), "dos": (209_590, 90_370),
                     "scan": (210_747, 90_330), "phy": (210_274, 90_496)},
}
MAKE_MASKS = Path(__file__).resolve().parent / "make_split_masks.py"
WINDOW, STRIDE = 50, 25
N_PAIRS = 2000
N_ROWS_CHECK = 2000
LEAKAGE_RATIO_LIMIT = 2.0
STREAM_CHUNK = 50_000

failures: list[str] = []


def report(ok: bool | None, msg: str) -> None:
    tag = "INFO" if ok is None else ("PASS" if ok else "FAIL")
    if tag == "FAIL":
        failures.append(msg)
    print(f"[{tag}] {msg}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def masks_regenerate(meta_path: Path, masks_dir: Path) -> bool:
    """Run make_split_masks.py with default settings and compare the masks byte for byte."""
    with tempfile.TemporaryDirectory() as tmp:
        run = subprocess.run([sys.executable, str(MAKE_MASKS), "--meta", str(meta_path),
                              "--output_dir", tmp], capture_output=True, text=True)
        if run.returncode:
            print(run.stderr)
            return False
        return all((Path(tmp) / name).read_bytes() == (masks_dir / name).read_bytes()
                   for name in (TRAIN_MASK_FILE, TEST_MASK_FILE))


def read_npy_header(f):
    """Shape and dtype of an open ``.npy`` stream (public numpy API, numpy 1.x and 2.x)."""
    version = np.lib.format.read_magic(f)
    reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
              else np.lib.format.read_array_header_2_0)
    shape, _, dtype = reader(f)
    return shape, dtype


def npz_header(path: Path):
    with zipfile.ZipFile(path) as zf, zf.open("windows.npy") as f:
        return read_npy_header(f)


def stream_windows(path: Path, wanted: np.ndarray) -> dict[int, np.ndarray]:
    """Read the windows at the (sorted, unique) indices ``wanted`` in one pass."""
    out, wanted = {}, np.unique(wanted)
    with zipfile.ZipFile(path) as zf, zf.open("windows.npy") as f:
        shape, dtype = read_npy_header(f)
        per = int(np.prod(shape[1:])) * np.dtype(dtype).itemsize
        pos, k = 0, 0
        while k < len(wanted) and pos < shape[0]:
            n = min(STREAM_CHUNK, shape[0] - pos)
            block = np.frombuffer(f.read(n * per), dtype=dtype).reshape((n,) + tuple(shape[1:]))
            while k < len(wanted) and wanted[k] < pos + n:
                out[int(wanted[k])] = block[wanted[k] - pos].copy()
                k += 1
            pos += n
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Verify the released ORAN-TWIN-IDS files.")
    p.add_argument("--data_dir", type=Path, default=None,
                   help="directory holding the released files under their release names")
    p.add_argument("--strong_csv", type=Path)
    p.add_argument("--stealth_csv", type=Path)
    p.add_argument("--windows", type=Path, help="windows_scaled.npz")
    p.add_argument("--meta", type=Path, help="windows_meta.csv")
    p.add_argument("--masks_dir", type=Path)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    d = args.data_dir
    paths = {
        "real_strong": args.strong_csv or (d / "real_strong.csv" if d else None),
        "real_stealth": args.stealth_csv or (d / "real_stealth.csv" if d else None),
    }
    windows = args.windows or (d / "windows_scaled.npz" if d else None)
    meta_path = args.meta or (d / "windows_meta.csv" if d else None)
    masks_dir = args.masks_dir or d
    rng = np.random.RandomState(args.seed)

    meta = None
    if meta_path and meta_path.is_file():
        meta = pd.read_csv(meta_path, dtype={"imsi": str, "reservation_id": str})
        cols = ["imsi", "reservation_id", "cluster", "start_timestamp", "label"]
        report(list(meta.columns) == cols, f"windows_meta.csv columns {list(meta.columns)}")
        report(len(meta) == EXPECTED_WINDOWS, f"windows_meta.csv rows {len(meta):,}")
        report(meta["cluster"].notna().all(), "every window has a cluster")
        report(None, f"windows per cluster {meta['cluster'].value_counts().sort_index().to_dict()}")
        ts = pd.to_datetime(meta["start_timestamp"], format="ISO8601")
        same = ((meta["reservation_id"].values[1:] == meta["reservation_id"].values[:-1])
                & (meta["imsi"].values[1:] == meta["imsi"].values[:-1]))
        report(bool((np.diff(ts.values.astype("int64"))[same] > 0).all()),
               "start times increase within every (reservation, IMSI) group")

    train = None
    if masks_dir and (masks_dir / TRAIN_MASK_FILE).is_file():
        for name, digest in MASK_SHA256.items():
            report(sha256(masks_dir / name) == digest, f"{name} SHA-256 {sha256(masks_dir / name)}")
        train = np.load(masks_dir / TRAIN_MASK_FILE, allow_pickle=False)
        test = np.load(masks_dir / TEST_MASK_FILE, allow_pickle=False)
        report(train.dtype == bool and test.dtype == bool, "masks are boolean")
        report(not (train & test).any() and (train | test).all(), "masks disjoint and complementary")
        report(None, f"train {train.sum():,} ({100 * train.mean():.1f}%) | test {test.sum():,}")
        if meta is not None and len(meta) == len(train):
            both = set(meta.reservation_id[train]) & set(meta.reservation_id[test])
            report(not both, f"reservations in both partitions: {len(both)}")
            report(masks_regenerate(meta_path, masks_dir),
                   "make_split_masks.py (default settings) regenerates both masks byte for byte")

    win = {}
    if windows and windows.is_file():
        shape, dtype = npz_header(windows)
        report(tuple(shape[1:]) == (WINDOW, 25) and np.dtype(dtype) == np.float32,
               f"windows_scaled.npz shape {shape} dtype {dtype}")
        report(shape[0] == EXPECTED_WINDOWS, f"windows_scaled.npz rows {shape[0]:,}")
        wanted = []
        pairs = boundaries = np.array([], dtype=np.int64)
        if meta is not None and len(meta) == shape[0]:
            pairs = rng.choice(np.flatnonzero(same), min(N_PAIRS, int(same.sum())), replace=False)
            boundaries = np.flatnonzero(~same)  # last window of a group, first of the next
            wanted += [pairs, pairs + 1, boundaries, boundaries + 1]
        rows = np.sort(rng.choice(shape[0], min(N_ROWS_CHECK, shape[0]), replace=False))
        wanted.append(rows)
        win = stream_windows(windows, np.concatenate(wanted))
        if len(pairs):
            ov = WINDOW - STRIDE

            def overlap(k):
                return np.array_equal(win[k][STRIDE:], win[k + 1][:ov])

            inside = np.mean([overlap(k) for k in pairs])
            report(inside == 1.0, f"meta/NPZ alignment: {100 * inside:.1f}% of sampled consecutive "
                                  f"same-group windows share their {ov} overlapping steps")
            across = np.mean([overlap(k) for k in boundaries]) if len(boundaries) else 0.0
            report(across < 0.01, f"meta/NPZ alignment: {100 * across:.1f}% of the {len(boundaries):,} "
                                  "group boundaries in the metadata look like overlapping windows "
                                  "in the NPZ (expected ~0%)")

    targets = set(INJECTION_TARGETS.values())
    for name, path in paths.items():
        if path is None:
            continue
        if not path.is_file():
            report(False, f"{path.name} not found in {path.parent} (unzip {path.stem}.zip first)")
            continue
        df = pd.read_csv(path, float_precision="round_trip")
        report(list(df.columns) == CSV_COLUMNS, f"{name}: 77-column schema")
        report(len(df) == EXPECTED_WINDOWS, f"{name}: rows {len(df):,}")
        report(not df[FEATURE_COLUMNS].isna().any().any(), f"{name}: no NaN")
        counts = df["attack_type"].value_counts().to_dict()
        report(counts == EXPECTED_CLASSES[name], f"{name}: class counts {counts}")
        report(bool(((df["attack_type"] != "benign").astype(int) == df["label"]).all()),
               f"{name}: label consistent with attack_type")
        if train is not None and len(train) == len(df):
            split = {c: (int(n_tr), int(n_te)) for c, n_tr, n_te in zip(
                EXPECTED_SPLIT_CLASSES[name],
                *[[int((df["attack_type"][m] == c).sum()) for c in EXPECTED_SPLIT_CLASSES[name]]
                  for m in (train, ~train)])}
            report(split == EXPECTED_SPLIT_CLASSES[name], f"{name}: (train, test) class counts {split}")
        if win:
            idx = np.array([r for r in rows if df.at[r, "label"] == 0])
            stats = window_statistics(np.stack([win[r] for r in idx]))
            ref = df.loc[idx, FEATURE_COLUMNS]
            # means/stds are stored as float32 text, p95 as float64 text
            equal = all(np.array_equal(stats[c].to_numpy(),
                                       ref[c].to_numpy(np.float64 if c.endswith("_p95") else np.float32))
                        for c in FEATURE_COLUMNS)
            report(equal, f"{name}: NPZ/CSV alignment on {len(idx)} sampled benign rows")
        att = df["label"] == 1
        worst = (None, 0.0)
        for c in FEATURE_COLUMNS:
            if c.endswith("_p95") and raw_index(c) not in targets:
                ub = df.loc[~att, c].round(6).nunique()
                ratio = df.loc[att, c].round(6).nunique() / max(ub, 1)
                if ratio > worst[1]:
                    worst = (c, ratio)
        report(worst[1] <= LEAKAGE_RATIO_LIMIT, f"{name}: non-target leakage gate, max distinct-value "
                                                f"ratio attack/benign {worst[1]:.2f} ({worst[0]})")
    print(f"\n{len(failures)} check(s) failed" if failures else "\nall checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
