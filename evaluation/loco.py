#!/usr/bin/env python3
"""Leave-one-cluster-out generalisation (paper Table 7).

For each Colosseum cluster, XGBoost and LightGBM are trained on the windows of
the other two clusters and tested on the held-out one (binary F1 and weighted
multi-class F1). The in-distribution reference (TRTR) is computed in the same
run with the same model settings on the reservation-level split masks.
gap = TRTR F1 - mean LOCO F1. Clusters are read from ``windows_meta.csv``, which
must be row-aligned with the variant CSV.

Models, loading and scaling are shared with every other evaluation script:
``boosting_models`` (random_state=42), ``load_variant`` and a StandardScaler
fitted on the training rows of each fold (``scale_partition``).

Example::

    python evaluation/loco.py --dataset data/real_stealth.csv \
        --meta data/windows_meta.csv --masks_dir data --output_dir results/loco
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oran_twin_ids.cli import require_file, setup_logging, write_run_info  # noqa: E402
from oran_twin_ids.data import load_meta, load_variant, scale_partition  # noqa: E402
from oran_twin_ids.evaluation import boosting_models  # noqa: E402
from oran_twin_ids.features import FEATURE_COLUMNS  # noqa: E402
from oran_twin_ids.splits import load_masks  # noqa: E402

log = logging.getLogger("loco")

def evaluate(X, y_bin, y_multi, train_idx, test_idx, tag) -> dict:
    if np.intersect1d(train_idx, test_idx).size or not (len(train_idx) and len(test_idx)):
        raise AssertionError(f"{tag}: invalid partition")
    X_tr, X_te = scale_partition(X, train_idx, test_idx)
    out = {}
    for name, template in boosting_models().items():
        m = clone(template).fit(X_tr, y_bin[train_idx])
        out[(name, "binary")] = f1_score(y_bin[test_idx], m.predict(X_te), zero_division=0)
        m = clone(template).fit(X_tr, y_multi[train_idx])
        out[(name, "multiclass")] = f1_score(y_multi[test_idx], m.predict(X_te),
                                             average="weighted", zero_division=0)
        log.info("  %-20s %-9s binary %.4f | multi %.4f", tag, name,
                 out[(name, "binary")], out[(name, "multiclass")])
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Leave-one-cluster-out evaluation (Table 7).")
    p.add_argument("--dataset", type=Path, required=True, help="variant CSV (paper: real_stealth)")
    p.add_argument("--meta", type=Path, required=True, help="windows_meta.csv (row-aligned)")
    p.add_argument("--masks_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, default=Path("results/loco"))
    args = p.parse_args(argv)
    setup_logging()
    require_file(p, args.dataset, "--dataset")
    require_file(p, args.meta, "--meta")

    df = load_variant(args.dataset)
    meta = load_meta(args.meta, n_rows=len(df))
    if (meta["label"].to_numpy() != 0).any():
        log.warning("windows_meta.csv has non-zero point labels")
    train_mask, test_mask = load_masks(args.masks_dir, n_rows=len(df))
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y_bin = df["label"].to_numpy(dtype=np.int8)
    le = LabelEncoder()
    y_multi = le.fit_transform(df["attack_type"]).astype(np.int8)
    clusters = meta["cluster"].to_numpy()
    del df

    rows = []
    log.info("TRTR baseline (reservation-level split)")
    for (model, task), f1 in evaluate(X, y_bin, y_multi, np.flatnonzero(train_mask),
                                      np.flatnonzero(test_mask), "TRTR").items():
        rows.append(dict(model=model, task=task, fold="TRTR_baseline", f1=f1))
    for cl in sorted(pd.unique(clusters)):
        tr, te = np.flatnonzero(clusters != cl), np.flatnonzero(clusters == cl)
        log.info("Hold out %s: train %s | test %s", cl, f"{len(tr):,}", f"{len(te):,}")
        for (model, task), f1 in evaluate(X, y_bin, y_multi, tr, te,
                                          f"holdout_{cl}").items():
            rows.append(dict(model=model, task=task, fold=f"holdout_{cl}", f1=f1))

    res = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.output_dir / "loco_results.csv", index=False)
    for task in ("binary", "multiclass"):
        piv = res[res.task == task].pivot(index="model", columns="fold", values="f1")
        hold = [c for c in piv.columns if c.startswith("holdout_")]
        piv["loco_mean"] = piv[hold].mean(axis=1)
        piv["gap"] = piv["TRTR_baseline"] - piv["loco_mean"]
        piv = piv[hold + ["loco_mean", "TRTR_baseline", "gap"]]
        piv.to_csv(args.output_dir / f"loco_summary_{task}.csv")
        print(f"\nLOCO - {task} F1\n{piv.round(4).to_string()}")
    write_run_info(args.output_dir, args)


if __name__ == "__main__":
    main()
