#!/usr/bin/env python3
"""Detection under realistic attack prevalence (paper Fig. 4; Table S9, tabular rows).

XGBoost and LightGBM are trained once on the full training partition (about
25 % attacks). The test partition is then evaluated as is and after randomly
removing attack windows (seed 42; benign windows untouched) until attacks make
up 10 %, 5 % and 1 %. Reported: PR-AUC (average precision), F1 at threshold
0.5, and precision at recall >= 0.90 / 0.95 (highest precision among operating
points meeting the recall target).

Example::

    python evaluation/prevalence_pr_auc.py --stealth_csv data/real_stealth.csv \
        --strong_csv data/real_strong.csv --masks_dir data --output_dir results/prevalence
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oran_twin_ids.cli import require_file, setup_logging, write_run_info  # noqa: E402
from oran_twin_ids.data import load_variant, make_split  # noqa: E402
from oran_twin_ids.evaluation import (PREVALENCES, boosting_models,  # noqa: E402
                                      downsample_attacks, precision_at_recall)

log = logging.getLogger("prevalence")


def evaluate_scores(y_test, scores) -> list[dict]:
    """Metrics at every prevalence level for one model's test scores."""
    rows = []
    for frac in PREVALENCES:
        idx = slice(None) if frac is None else downsample_attacks(y_test, frac)
        y, s = y_test[idx], scores[idx]
        rows.append({
            "prevalence": "original" if frac is None else f"{frac:.0%}",
            "actual_prevalence": round(float(y.mean()), 4),
            "n_test": len(y), "n_attacks": int(y.sum()),
            "pr_auc": average_precision_score(y, s),
            "f1_at_0.5": f1_score(y, (s >= 0.5).astype(int), zero_division=0),
            "prec_at_recall_0.90": precision_at_recall(y, s, 0.90),
            "prec_at_recall_0.95": precision_at_recall(y, s, 0.95),
            "_curve": precision_recall_curve(y, s),
        })
    return rows


def plot_curves(rows: list[dict], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for r, ls in zip(rows, ["-", "--", "-.", ":"]):
        precision, recall, _ = r["_curve"]
        ax.plot(recall, precision, linestyle=ls, linewidth=2,
                label=f"{r['prevalence']} (PR-AUC={r['pr_auc']:.3f})")
    ax.axvline(0.9, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Recall = 0.90")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.05), title=title)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="PR-AUC under reduced attack prevalence.")
    p.add_argument("--strong_csv", type=Path)
    p.add_argument("--stealth_csv", type=Path)
    p.add_argument("--masks_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, default=Path("results/prevalence"))
    args = p.parse_args(argv)
    setup_logging()
    if not (args.strong_csv or args.stealth_csv):
        p.error("give --strong_csv and/or --stealth_csv")
    require_file(p, args.strong_csv, "--strong_csv")
    require_file(p, args.stealth_csv, "--stealth_csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    table = []
    for variant, path in (("real_stealth", args.stealth_csv), ("real_strong", args.strong_csv)):
        if path is None:
            continue
        split = make_split(load_variant(path), args.masks_dir)
        print(f"{variant}: test windows {len(split.y_test_bin):,}, "
              f"attacks {int(split.y_test_bin.sum()):,}")
        for name, template in boosting_models().items():
            model = clone(template).fit(split.X_train, split.y_train_bin)
            rows = evaluate_scores(split.y_test_bin, model.predict_proba(split.X_test)[:, 1])
            plot_curves(rows, f"{name} - {variant}", args.output_dir / f"pr_curves_{name}_{variant}.png")
            table += [{"variant": variant, "model": name,
                       **{k: v for k, v in r.items() if k != "_curve"}} for r in rows]
    res = pd.DataFrame(table)
    res.to_csv(args.output_dir / "prevalence_results.csv", index=False)
    print(res.round(4).to_string(index=False))
    write_run_info(args.output_dir, args)


if __name__ == "__main__":
    main()
