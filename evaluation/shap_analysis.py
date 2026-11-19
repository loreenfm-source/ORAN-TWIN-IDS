#!/usr/bin/env python3
"""TreeSHAP attributions for XGBoost and LightGBM (Figs. S8-S9, Table 10).

Binary models are trained on the reservation-level training partition;
SHAP values are computed on a 5,000-window test subsample that preserves the
test attack ratio (seed 42). Feature names are the human-readable aliases of
``oran_twin_ids.features.FEATURE_ALIASES``, mapped one-to-one onto the 75 CSV
columns.

Outputs per variant and model: ``shap_bar_<model>_<variant>.png``,
``shap_beeswarm_<model>_<variant>.png`` and ``shap_ranking_<model>_<variant>.csv``
(mean |SHAP| of all 75 features, ranked).

Example::

    python evaluation/shap_analysis.py --stealth_csv data/real_stealth.csv \
        --masks_dir data --output_dir results/shap
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oran_twin_ids.cli import require_file, setup_logging, write_run_info  # noqa: E402
from oran_twin_ids.data import load_variant, make_split  # noqa: E402
from oran_twin_ids.evaluation import SEED, boosting_models  # noqa: E402
from oran_twin_ids.features import FEATURE_COLUMNS, readable_name  # noqa: E402

log = logging.getLogger("shap")
SAMPLE_SIZE = 5000
TOP_K = 20


def stratified_sample(y: np.ndarray, size: int, seed: int = SEED) -> np.ndarray:
    """Indices of ``size`` rows keeping the attack ratio of ``y`` (sorted)."""
    rng = np.random.RandomState(seed)
    size = min(size, len(y))
    benign, attack = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    n_attack = int(size * len(attack) / len(y))
    return np.sort(np.concatenate([rng.choice(benign, size - n_attack, replace=False),
                                   rng.choice(attack, n_attack, replace=False)]))


def run(variant: str, csv_path: Path, masks_dir: Path, out: Path) -> None:
    import shap

    split = make_split(load_variant(csv_path), masks_dir)
    names = [readable_name(c) for c in FEATURE_COLUMNS]
    X_train = pd.DataFrame(split.X_train, columns=names)
    sel = stratified_sample(split.y_test_bin, SAMPLE_SIZE)
    X_shap = pd.DataFrame(split.X_test[sel], columns=names)
    for model_name, template in boosting_models().items():
        log.info("[%s] training %s", variant, model_name)
        model = clone(template).fit(X_train, split.y_train_bin)
        values = shap.TreeExplainer(model).shap_values(X_shap)
        if isinstance(values, list):  # some versions return one array per class
            values = values[1]
        for kind, plot_type in (("bar", "bar"), ("beeswarm", "dot")):
            plt.figure(figsize=(10, 8))
            shap.summary_plot(values, X_shap, plot_type=plot_type, max_display=TOP_K, show=False)
            plt.title(f"{model_name} - SHAP ({variant})", fontsize=12)
            plt.tight_layout()
            plt.savefig(out / f"shap_{kind}_{model_name}_{variant}.png", dpi=300, bbox_inches="tight")
            plt.close()
        ranking = pd.DataFrame({"feature": names, "column": FEATURE_COLUMNS,
                                "mean_abs_shap": np.abs(values).mean(axis=0)})
        ranking = ranking.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        ranking.index += 1
        ranking.to_csv(out / f"shap_ranking_{model_name}_{variant}.csv", index_label="rank")
        print(f"\n{model_name} on {variant} - top {TOP_K} by mean |SHAP|\n"
              f"{ranking.head(TOP_K).round(4).to_string()}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="TreeSHAP analysis (Figs. S8-S9).")
    p.add_argument("--strong_csv", type=Path)
    p.add_argument("--stealth_csv", type=Path)
    p.add_argument("--masks_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, default=Path("results/shap"))
    args = p.parse_args(argv)
    setup_logging()
    if not (args.strong_csv or args.stealth_csv):
        p.error("give --strong_csv and/or --stealth_csv")
    require_file(p, args.strong_csv, "--strong_csv")
    require_file(p, args.stealth_csv, "--stealth_csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, path in (("real_strong", args.strong_csv), ("real_stealth", args.stealth_csv)):
        if path is not None:
            run(name, path, args.masks_dir, args.output_dir)
    write_run_info(args.output_dir, args)


if __name__ == "__main__":
    main()
