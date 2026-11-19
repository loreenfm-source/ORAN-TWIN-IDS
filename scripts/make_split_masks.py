#!/usr/bin/env python3
"""Reservation-level train/test masks from ``windows_meta.csv``.

Within each (cluster, slicing, scheduling) configuration cell, the first
``--train_frac`` of reservations (identifier order) form the training
partition; all windows of the remaining reservations form the test partition.
No reservation appears in both, and every configuration cell contributes to
both partitions. ``--strata cluster`` gives the earlier per-cluster split.

Slicing and scheduling are attached per reservation from ``--configuration``
(default ``metadata/reservation_configuration.csv``, taken from the source
directory layout ``cluster_*/slicing_*/scheduling_*/RESERVATION-*``).

Outputs ``split_train_mask.npy`` and ``split_test_mask.npy`` (boolean, one entry
per window, row-aligned with ``windows_meta.csv``) and prints per-cell
reservation counts, the partition sizes, and the cluster and configuration
mixes. Existing masks in ``--output_dir`` are only replaced with ``--overwrite``.

If the metadata file has no ``cluster`` column (e.g. a file written by an
earlier version of Stage 2 with columns ``reservation_id, imsi, start_ts,
label``), clusters are attached from ``--cluster_map``; ``--write_meta`` then
saves the completed file in the released column layout.

Example::

    python scripts/make_split_masks.py --meta outputs/windows/windows_meta.csv \
        --output_dir outputs/windows
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from oran_twin_ids.data import META_COLUMNS  # noqa: E402
from oran_twin_ids.splits import (DEFAULT_STRATA, DEFAULT_TRAIN_FRAC,  # noqa: E402
                                  TEST_MASK_FILE, TRAIN_MASK_FILE, check_disjoint,
                                  reservation_split)

log = logging.getLogger("splits")
MIN_CELL_RESERVATIONS = 5


def complete_meta(meta: pd.DataFrame, cluster_map_path: Path) -> pd.DataFrame:
    """Add ``cluster`` (and rename legacy ``start_ts``) so ``meta`` has META_COLUMNS."""
    meta = meta.rename(columns={"start_ts": "start_timestamp"})
    if "cluster" not in meta.columns:
        cmap = json.loads(Path(cluster_map_path).read_text(encoding="utf-8"))
        meta["cluster"] = meta["reservation_id"].map(cmap)
        unmapped = meta.loc[meta["cluster"].isna(), "reservation_id"].unique()
        if len(unmapped):
            raise ValueError(f"{len(unmapped)} reservations missing from {cluster_map_path}")
    missing = set(META_COLUMNS) - set(meta.columns)
    if missing:
        raise ValueError(f"metadata lacks columns {sorted(missing)}")
    return meta[list(META_COLUMNS)]


def attach_configuration(meta: pd.DataFrame, config_path: Path) -> pd.DataFrame:
    """Return ``meta`` with ``slicing`` and ``scheduling`` columns from ``config_path``."""
    cfg = pd.read_csv(config_path, dtype=str).set_index("reservation_id")
    res = meta["reservation_id"]
    unmapped = res[~res.isin(cfg.index)].unique()
    if len(unmapped):
        raise ValueError(f"{len(unmapped)} reservations missing from {config_path}")
    if not (res.map(cfg["cluster"]) == meta["cluster"]).all():
        raise ValueError(f"cluster column disagrees with {config_path}")
    out = meta.copy()
    for col in ("slicing", "scheduling"):
        out[col] = res.map(cfg[col]).to_numpy()
    return out


def report(meta: pd.DataFrame, train: np.ndarray, strata: list[str]) -> None:
    """Print per-cell reservation counts, partition sizes and mixes."""
    test = ~train
    res = meta["reservation_id"].to_numpy()
    cells = (meta.assign(part=np.where(train, "train", "test"))
             .groupby(["cluster", "slicing", "scheduling", "part"])["reservation_id"].nunique()
             .unstack("part", fill_value=0).reindex(columns=["train", "test"], fill_value=0))
    cells["total"] = cells.sum(axis=1)
    small = cells["total"] < MIN_CELL_RESERVATIONS
    cells["flag"] = np.where(small, f"< {MIN_CELL_RESERVATIONS} reservations", "")
    print(f"split cells: {' x '.join(strata)}")
    print(f"reservations per configuration cell:\n{cells.to_string()}")
    print(f"cells with < {MIN_CELL_RESERVATIONS} reservations: {int(small.sum())}")

    overlap = len(set(res[train]) & set(res[test]))
    print(f"windows {len(meta):,} | train {train.sum():,} ({100 * train.mean():.1f}%) | "
          f"test {test.sum():,} ({100 * test.mean():.1f}%)")
    print(f"reservations: train {len(set(res[train]))} | test {len(set(res[test]))} | "
          f"overlap {overlap}")
    config = meta["slicing"] + "/" + meta["scheduling"]
    for title, col in (("cluster", meta["cluster"]), ("configuration", config)):
        mix = pd.DataFrame({part: col[m].value_counts(normalize=True)
                            for part, m in (("train", train), ("test", test))})
        print(f"{title} mix (% of windows):\n"
              f"{(mix.fillna(0).sort_index() * 100).round(1).to_string()}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Create reservation-level split masks.")
    p.add_argument("--meta", type=Path, required=True, help="windows_meta.csv")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--train_frac", type=float, default=DEFAULT_TRAIN_FRAC)
    p.add_argument("--strata", nargs="+", choices=list(DEFAULT_STRATA),
                   default=list(DEFAULT_STRATA), help="columns defining the split cells")
    p.add_argument("--cluster_map", type=Path,
                   default=REPO_ROOT / "metadata" / "reservation_cluster_map.json")
    p.add_argument("--configuration", type=Path,
                   default=REPO_ROOT / "metadata" / "reservation_configuration.csv")
    p.add_argument("--write_meta", action="store_true",
                   help="write the completed windows_meta.csv to --output_dir")
    p.add_argument("--overwrite", action="store_true",
                   help="replace masks already present in --output_dir")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for path in (args.meta, args.configuration):
        if not path.is_file():
            p.error(f"file not found: {path}")
    targets = [args.output_dir / TRAIN_MASK_FILE, args.output_dir / TEST_MASK_FILE]
    if not args.overwrite and any(t.exists() for t in targets):
        p.error(f"masks already exist in {args.output_dir}; pass --overwrite to replace them")

    meta = complete_meta(pd.read_csv(args.meta, dtype={"imsi": str, "reservation_id": str}),
                         args.cluster_map)
    full = attach_configuration(meta, args.configuration)
    train = reservation_split(full, args.train_frac, tuple(args.strata))
    check_disjoint(full["reservation_id"].to_numpy(), train)  # raises on any overlap
    report(full, train, args.strata)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.write_meta:
        out_meta = args.output_dir / "windows_meta.csv"
        if out_meta.resolve() == args.meta.resolve():
            p.error("--write_meta would overwrite the input; choose another --output_dir")
        meta.to_csv(out_meta, index=False)
        log.info("wrote %s", out_meta)
    np.save(targets[0], train)
    np.save(targets[1], ~train)
    log.info("wrote %s and %s", *targets)


if __name__ == "__main__":
    main()
