#!/usr/bin/env python3
"""Injection-aligned feature-group ablation (supplementary Table S6).

Groups (raw signal indices; each contributes its mean/std/p95 columns):
A = bitrate/PRB {f0, f1, f14, f15, f16} (DoS), B = packets/errors
{f8, f9, f10, f11} (Scan), C = radio quality {f2, f3, f13, f17} (PHY).
XGBoost and LightGBM are retrained on the reservation-level split after
removing each group and each combination; the binary F1 and the per-class
recall of the multi-class model are reported.

Example::

    python evaluation/feature_group_ablation.py --dataset data/real_stealth.csv \
        --masks_dir data --output_dir results/ablation
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oran_twin_ids.cli import require_file, setup_logging, write_run_info  # noqa: E402
from oran_twin_ids.data import load_variant, make_split  # noqa: E402
from oran_twin_ids.evaluation import boosting_models  # noqa: E402
from oran_twin_ids.features import ABLATION_GROUPS, FEATURE_COLUMNS, columns_for  # noqa: E402

log = logging.getLogger("ablation")


def conditions() -> list[tuple[str, ...]]:
    """(), (A,), (B,), (C,), (A,B), (A,C), (B,C), (A,B,C)."""
    groups = sorted(ABLATION_GROUPS)
    return [c for r in range(len(groups) + 1) for c in itertools.combinations(groups, r)]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Feature-group ablation (Table S6).")
    p.add_argument("--dataset", type=Path, required=True, help="variant CSV (paper: real_stealth)")
    p.add_argument("--masks_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, default=Path("results/ablation"))
    args = p.parse_args(argv)
    setup_logging()
    require_file(p, args.dataset, "--dataset")

    df = load_variant(args.dataset)
    rows = []
    for removed in conditions():
        drop = set(columns_for([i for g in removed for i in ABLATION_GROUPS[g]], FEATURE_COLUMNS))
        keep = [c for c in FEATURE_COLUMNS if c not in drop]
        label = "none" if not removed else "-" + "-".join(removed)
        log.info("Condition %s: %d features", label, len(keep))
        split = make_split(df, args.masks_dir, columns=keep)
        for name, template in boosting_models().items():
            m_bin = clone(template).fit(split.X_train, split.y_train_bin)
            bin_f1 = f1_score(split.y_test_bin, m_bin.predict(split.X_test), zero_division=0)
            m_multi = clone(template).fit(split.X_train, split.y_train_multi)
            recalls = recall_score(split.y_test_multi, m_multi.predict(split.X_test),
                                   average=None, labels=range(len(split.classes)), zero_division=0)
            row = {"removed_groups": label, "n_features": len(keep), "model": name,
                   "binary_f1": bin_f1}
            row.update({f"recall_{c}": r for c, r in zip(split.classes, recalls)})
            rows.append(row)
            log.info("  %s binary F1 %.4f | recalls %s", name, bin_f1,
                     dict(zip(split.classes, recalls.round(4))))
    res = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.output_dir / "feature_group_ablation.csv", index=False)
    print(res.round(4).to_string(index=False))
    write_run_info(args.output_dir, args)


if __name__ == "__main__":
    main()
