#!/usr/bin/env python3
"""Single-feature decision stumps (shortcut check, supplementary Table S7).

One depth-1 decision tree per statistical column is trained on a random
subsample of 50,000 training windows and evaluated (binary F1) on 20,000 test
windows, both drawn (seed 42) within the reservation-level partitions. The
features are scaled as in every other experiment (``make_split``: StandardScaler
fitted on the training partition). The F1 of predicting
"attack" for every test window is printed as the trivial reference.

Example::

    python evaluation/decision_stumps.py --stealth_csv data/real_stealth.csv \
        --strong_csv data/real_strong.csv --masks_dir data --output_dir results/stumps
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oran_twin_ids.cli import require_file, setup_logging, write_run_info  # noqa: E402
from oran_twin_ids.data import load_variant, make_split  # noqa: E402
from oran_twin_ids.evaluation import SEED  # noqa: E402
from oran_twin_ids.features import FEATURE_COLUMNS  # noqa: E402

log = logging.getLogger("stumps")
N_TRAIN, N_TEST = 50_000, 20_000


def stump_scores(csv_path: Path, masks_dir: Path, n_train: int = N_TRAIN,
                 n_test: int = N_TEST, seed: int = SEED) -> tuple[pd.DataFrame, float]:
    """Return per-column stump F1 (sorted) and the trivial all-attack F1."""
    split = make_split(load_variant(csv_path), masks_dir)
    rng = np.random.RandomState(seed)
    n_tr, n_te = len(split.y_train_bin), len(split.y_test_bin)
    tr = np.sort(rng.choice(n_tr, min(n_train, n_tr), replace=False))
    te = np.sort(rng.choice(n_te, min(n_test, n_te), replace=False))
    X_tr, X_te = split.X_train[tr], split.X_test[te]
    y_tr, y_te = split.y_train_bin[tr], split.y_test_bin[te]
    rows = []
    for i, col in enumerate(FEATURE_COLUMNS):
        stump = DecisionTreeClassifier(max_depth=1, random_state=seed).fit(X_tr[:, [i]], y_tr)
        rows.append({"feature": col,
                     "f1_binary": f1_score(y_te, stump.predict(X_te[:, [i]]), zero_division=0)})
    trivial = f1_score(y_te, np.ones_like(y_te), zero_division=0)
    res = pd.DataFrame(rows).sort_values("f1_binary", ascending=False, kind="stable")
    return res.reset_index(drop=True), trivial


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Decision-stump shortcut check (Table S7).")
    p.add_argument("--strong_csv", type=Path)
    p.add_argument("--stealth_csv", type=Path)
    p.add_argument("--masks_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, default=Path("results/stumps"))
    args = p.parse_args(argv)
    setup_logging()
    if not (args.strong_csv or args.stealth_csv):
        p.error("give --strong_csv and/or --stealth_csv")
    require_file(p, args.strong_csv, "--strong_csv")
    require_file(p, args.stealth_csv, "--stealth_csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, path in (("real_strong", args.strong_csv), ("real_stealth", args.stealth_csv)):
        if path is None:
            continue
        res, trivial = stump_scores(path, args.masks_dir)
        res.to_csv(args.output_dir / f"stumps_{name}.csv", index=False)
        print(f"\n{name}: trivial all-attack F1 = {trivial:.4f}\n"
              f"{res.head(12).round(4).to_string(index=False)}")
    write_run_info(args.output_dir, args)


if __name__ == "__main__":
    main()
