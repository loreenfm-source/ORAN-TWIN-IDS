#!/usr/bin/env python3
"""Split checks on one variant (XGBoost binary F1; paper: real_stealth).

1. Random reservation-level splits: for each seed, the reservations of every
   (cluster, slicing, scheduling) cell are shuffled (numpy ``default_rng(seed)``)
   and the first ``--train_frac`` of them form the training partition. The F1
   spread over seeds shows how much the released split's F1 depends on which
   reservations it happens to hold out.
2. Released split: F1 on the released masks, and the train/test window share,
   reservation counts and test F1 per slicing/scheduling configuration.

Slicing and scheduling are attached per reservation from ``--configuration``
(default ``metadata/reservation_configuration.csv``). Models, loading and
scaling are shared with every other evaluation script: ``boosting_models``
(random_state=42), ``load_variant`` and ``scale_partition``.

Example::

    python evaluation/split_checks.py --dataset data/real_stealth.csv \
        --meta data/windows_meta.csv --masks_dir data --output_dir results/split_checks
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from oran_twin_ids.cli import require_file, setup_logging, write_run_info  # noqa: E402
from oran_twin_ids.data import load_meta, load_variant, scale_partition  # noqa: E402
from oran_twin_ids.evaluation import boosting_models  # noqa: E402
from oran_twin_ids.features import FEATURE_COLUMNS  # noqa: E402
from oran_twin_ids.splits import DEFAULT_TRAIN_FRAC, check_disjoint, load_masks  # noqa: E402

log = logging.getLogger("split_checks")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Random-split spread and per-configuration F1.")
    p.add_argument("--dataset", type=Path, required=True, help="variant CSV (paper: real_stealth)")
    p.add_argument("--meta", type=Path, required=True, help="windows_meta.csv (row-aligned)")
    p.add_argument("--masks_dir", type=Path, required=True)
    p.add_argument("--configuration", type=Path,
                   default=REPO_ROOT / "metadata" / "reservation_configuration.csv")
    p.add_argument("--output_dir", type=Path, default=Path("results/split_checks"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="seeds of the random reservation-level splits")
    p.add_argument("--train_frac", type=float, default=DEFAULT_TRAIN_FRAC)
    args = p.parse_args(argv)
    setup_logging()
    for path, flag in ((args.dataset, "--dataset"), (args.meta, "--meta"),
                       (args.configuration, "--configuration")):
        require_file(p, path, flag)

    df = load_variant(args.dataset)
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy()
    del df
    meta = load_meta(args.meta, n_rows=len(y))
    cfg = pd.read_csv(args.configuration, dtype=str).set_index("reservation_id")
    if not (meta.reservation_id.map(cfg["cluster"]) == meta.cluster).all():
        raise ValueError(f"cluster column disagrees with {args.configuration}")
    res = meta.reservation_id.to_numpy()
    config = (meta.reservation_id.map(cfg["slicing"]) + "/"
              + meta.reservation_id.map(cfg["scheduling"])).to_numpy()
    cell = meta.cluster.to_numpy() + "/" + config
    report = {}

    def fit_predict(tr, te):
        X_tr, X_te = scale_partition(X, tr, te)
        return clone(boosting_models()["XGBoost"]).fit(X_tr, y[tr]).predict(X_te)

    log.info("Random reservation-level splits per configuration cell")
    rand = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        tr = np.zeros(len(res), bool)
        for c in sorted(set(cell)):
            ids = np.unique(res[cell == c])
            rng.shuffle(ids)
            tr |= np.isin(res, ids[:int(round(args.train_frac * len(ids)))])
        check_disjoint(res, tr)
        f1 = f1_score(y[~tr], fit_predict(tr, ~tr))
        rand.append(dict(seed=seed, train=int(tr.sum()), test=int((~tr).sum()), f1=f1))
        log.info("  seed %d: train %s | test %s | F1 %.4f", seed, f"{tr.sum():,}",
                 f"{(~tr).sum():,}", f1)
    f1s = np.array([r["f1"] for r in rand])
    report["random_splits"] = rand
    report["random_splits_mean"] = float(f1s.mean())
    report["random_splits_std"] = float(f1s.std(ddof=1)) if len(f1s) > 1 else float("nan")
    log.info("  mean %.4f +- %.4f (sample s.d.)", report["random_splits_mean"],
             report["random_splits_std"])

    log.info("Released split, per slicing/scheduling configuration")
    tr, te = load_masks(args.masks_dir, n_rows=len(y))
    check_disjoint(res, tr)
    pred = fit_predict(tr, te)
    report["released_split_f1"] = f1_score(y[te], pred)
    log.info("  released masks: train %s | test %s | F1 %.4f", f"{tr.sum():,}",
             f"{te.sum():,}", report["released_split_f1"])
    rows = []
    for k in sorted(set(config)):
        sel = config[te] == k
        rows.append(dict(configuration=k,
                         train_share=float((config[tr] == k).mean()),
                         test_share=float(sel.mean()),
                         train_reservations=len(set(res[tr & (config == k)])),
                         test_reservations=len(set(res[te & (config == k)])),
                         test_windows=int(sel.sum()),
                         f1=f1_score(y[te][sel], pred[sel]) if sel.any() else float("nan")))
    cdf = pd.DataFrame(rows)
    print(cdf.round(4).to_string(index=False))
    report["per_configuration"] = rows

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cdf.to_csv(out / "per_configuration.csv", index=False)
    pd.DataFrame(rand).to_csv(out / "random_splits.csv", index=False)
    (out / "split_checks.json").write_text(json.dumps(report, indent=2, default=float),
                                           encoding="utf-8")
    write_run_info(out, args)


if __name__ == "__main__":
    main()
