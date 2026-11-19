#!/usr/bin/env python3
"""Baseline classifiers on the released variants (paper Table 6; Tables S2-S5; Figs. S6-S7).

For every variant given, seven classifiers are trained on the reservation-level
training partition (StandardScaler fitted on training rows only) for the binary
task (benign vs attack) and the four-class task (benign, dos, phy, scan), and
evaluated on the test partition. Multi-class precision/recall/F1 are weighted
averages. A fresh model instance is used per (model, task), and confusion
matrices are drawn from exactly the fitted models that produce the tables.

Loading, scaling and seeds are shared with every other evaluation script
(``load_variant``, ``make_split``, ``random_state=42``).

Outputs per variant in ``--output_dir/<variant>/``: ``class_counts.csv``
(train/test windows per class), ``binary_results.csv``,
``multiclass_results.csv``, ``multiclass_perclass_reports/<model>.csv``,
``binary_confusion_matrices/<model>.png`` (+ ``.csv``) and
``multiclass_confusion_matrices/<model>.png`` (+ ``.csv``).

Example::

    python evaluation/baseline_classifiers.py --stealth_csv data/real_stealth.csv \
        --strong_csv data/real_strong.csv --masks_dir data --output_dir results/baselines
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oran_twin_ids.cli import require_file, setup_logging, write_run_info  # noqa: E402
from oran_twin_ids.data import load_variant, make_split  # noqa: E402
from oran_twin_ids.evaluation import baseline_models, save_confusion_matrix  # noqa: E402

log = logging.getLogger("baselines")


def evaluate_variant(name: str, csv_path: Path, masks_dir: Path, out_dir: Path,
                     model_names: list[str] | None) -> None:
    split = make_split(load_variant(csv_path), masks_dir)
    models = baseline_models()
    if model_names:
        models = {k: models[k] for k in model_names}
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = pd.DataFrame({part: pd.Series(y).map(dict(enumerate(split.classes))).value_counts()
                           for part, y in (("train", split.y_train_multi),
                                           ("test", split.y_test_multi))})
    counts.loc["total"] = counts.sum()
    counts.to_csv(out_dir / "class_counts.csv")
    log.info("[%s] class counts\n%s", name, counts.to_string())

    tasks = {
        "binary": (split.y_train_bin, split.y_test_bin, ["Benign", "Attack"], "binary"),
        "multiclass": (split.y_train_multi, split.y_test_multi, list(split.classes), "weighted"),
    }
    for task, (y_train, y_test, labels, average) in tasks.items():
        rows = []
        for model_name, template in models.items():
            log.info("[%s] %s %s", name, task, model_name)
            model = clone(template)
            t0 = time.perf_counter()
            model.fit(split.X_train, y_train)
            pred = model.predict(split.X_test)
            rows.append({
                "model": model_name,
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, average=average, zero_division=0),
                "recall": recall_score(y_test, pred, average=average, zero_division=0),
                "f1_score": f1_score(y_test, pred, average=average, zero_division=0),
                "fit_predict_seconds": round(time.perf_counter() - t0, 1),
            })
            log.info("    acc=%.4f f1=%.4f", rows[-1]["accuracy"], rows[-1]["f1_score"])
            cm = confusion_matrix(y_test, pred)
            cm_dir = out_dir / f"{task}_confusion_matrices"
            cm_dir.mkdir(exist_ok=True)
            pd.DataFrame(cm, index=labels, columns=labels).to_csv(cm_dir / f"{model_name}.csv")
            save_confusion_matrix(cm, labels, f"{model_name} {task} confusion matrix ({name})",
                                  cm_dir / f"{model_name}.png")
            if task == "multiclass":
                report = classification_report(y_test, pred, target_names=labels,
                                               output_dict=True, zero_division=0)
                rep_dir = out_dir / "multiclass_perclass_reports"
                rep_dir.mkdir(exist_ok=True)
                pd.DataFrame(report).transpose().to_csv(rep_dir / f"{model_name}.csv")
        results = pd.DataFrame(rows)
        results.to_csv(out_dir / f"{task}_results.csv", index=False)
        print(f"\n{name} - {task}\n{results.round(4).to_string(index=False)}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Seven baseline classifiers (Table 6, S2-S5).")
    p.add_argument("--strong_csv", type=Path, help="real_strong.csv")
    p.add_argument("--stealth_csv", type=Path, help="real_stealth.csv")
    p.add_argument("--masks_dir", type=Path, required=True,
                   help="directory with split_train_mask.npy and split_test_mask.npy")
    p.add_argument("--output_dir", type=Path, default=Path("results/baselines"))
    p.add_argument("--models", nargs="+", choices=list(baseline_models()),
                   help="subset of models (default: all seven)")
    args = p.parse_args(argv)
    setup_logging()
    if not (args.strong_csv or args.stealth_csv):
        p.error("give --strong_csv and/or --stealth_csv")
    require_file(p, args.strong_csv, "--strong_csv")
    require_file(p, args.stealth_csv, "--stealth_csv")
    for name, path in (("real_strong", args.strong_csv), ("real_stealth", args.stealth_csv)):
        if path is not None:
            evaluate_variant(name, path, args.masks_dir, args.output_dir / name, args.models)
    write_run_info(args.output_dir, args)


if __name__ == "__main__":
    main()
