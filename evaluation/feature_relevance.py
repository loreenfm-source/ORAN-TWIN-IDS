#!/usr/bin/env python3
"""Feature relevance and class separability on one variant (Figs. 2, 3, S2-S5).

* Pearson correlation of each feature with the binary label (constant columns
  are excluded because the coefficient is undefined for them);
* mutual information (``mutual_info_classif``, seed 42) on all windows;
* Random Forest (200 trees, seed 42) Gini importance, fitted on all windows -
  a descriptive analysis, not a generalisation estimate;
* inter-feature Pearson correlation heatmap;
* t-SNE (perplexity 30, 1000 iterations) and UMAP (30 neighbours,
  min_dist 0.1) of a random 10,000-window sample (seed 42).

``--skip`` omits expensive parts. MI and the Random Forest on all 3.6 M windows
take hours on a CPU.

Example::

    python evaluation/feature_relevance.py --dataset data/real_stealth.csv \
        --output_dir results/feature_relevance
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
import seaborn as sns  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oran_twin_ids.cli import require_file, setup_logging, write_run_info  # noqa: E402
from oran_twin_ids.data import load_variant  # noqa: E402
from oran_twin_ids.evaluation import SEED  # noqa: E402
from oran_twin_ids.features import FEATURE_COLUMNS  # noqa: E402

log = logging.getLogger("relevance")
TOP_K = 20
EMBED_SAMPLE = 10_000
PARTS = ("pearson", "mi", "rf", "heatmap", "tsne", "umap")


def bar_plot(df: pd.DataFrame, x: str, title: str, path: Path) -> None:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=x, y="feature", data=df.head(TOP_K))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def scatter_embedding(emb: np.ndarray, labels: np.ndarray, title: str, path: Path) -> None:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=labels, palette="Set1", alpha=0.7, s=30)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Feature relevance and projections.")
    p.add_argument("--dataset", type=Path, required=True, help="variant CSV (paper: real_stealth)")
    p.add_argument("--output_dir", type=Path, default=Path("results/feature_relevance"))
    p.add_argument("--skip", nargs="+", choices=PARTS, default=[], help="parts to skip")
    args = p.parse_args(argv)
    setup_logging()
    require_file(p, args.dataset, "--dataset")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    df = load_variant(args.dataset)
    X, y = df[FEATURE_COLUMNS], df["label"]
    variable = [c for c in FEATURE_COLUMNS if X[c].std() > 0]
    log.info("%d of %d feature columns vary", len(variable), len(FEATURE_COLUMNS))

    if "pearson" not in args.skip:
        corr = X[variable].corrwith(y)
        res = pd.DataFrame({"feature": corr.index, "pearson": corr.values,
                            "abs_pearson": corr.abs().values}).sort_values("abs_pearson", ascending=False)
        res.to_csv(out / "feature_label_correlation.csv", index=False)
        bar_plot(res, "abs_pearson", "Top features by |Pearson correlation| with label",
                 out / "feature_label_correlation.png")
        print("Top |Pearson|:\n", res.head(10).round(4).to_string(index=False))

    if "mi" not in args.skip:
        from sklearn.feature_selection import mutual_info_classif
        log.info("Mutual information on %s windows", f"{len(X):,}")
        mi = mutual_info_classif(X, y, random_state=SEED)
        res = pd.DataFrame({"feature": FEATURE_COLUMNS, "mutual_information": mi}) \
            .sort_values("mutual_information", ascending=False)
        res.to_csv(out / "feature_mutual_information.csv", index=False)
        bar_plot(res, "mutual_information", "Top features by mutual information",
                 out / "feature_mutual_information.png")
        print("Top MI:\n", res.head(10).round(4).to_string(index=False))

    if "rf" not in args.skip:
        from sklearn.ensemble import RandomForestClassifier
        log.info("Random Forest importance on all windows")
        rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1).fit(X, y)
        res = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": rf.feature_importances_}) \
            .sort_values("importance", ascending=False)
        res.to_csv(out / "rf_feature_importance.csv", index=False)
        bar_plot(res, "importance", "Top Random Forest feature importances",
                 out / "rf_feature_importance.png")
        print("Top RF importance:\n", res.head(10).round(4).to_string(index=False))

    if "heatmap" not in args.skip:
        plt.figure(figsize=(12, 10))
        sns.heatmap(X.corr(), cmap="coolwarm", center=0, cbar=True)  # constant columns stay blank
        plt.title("Feature correlation heatmap")
        plt.tight_layout()
        plt.savefig(out / "feature_correlation_heatmap.png", dpi=300)
        plt.close()

    if not {"tsne", "umap"} <= set(args.skip):
        sample = df.sample(min(EMBED_SAMPLE, len(df)), random_state=SEED)
        Xs = sample[FEATURE_COLUMNS].to_numpy(np.float32)
        ys = sample["attack_type"].to_numpy()
        if "tsne" not in args.skip:
            from sklearn.manifold import TSNE
            emb = TSNE(n_components=2, perplexity=30, max_iter=1000,
                       random_state=SEED).fit_transform(Xs)
            scatter_embedding(emb, ys, "t-SNE projection", out / "tsne_visualization.png")
        if "umap" not in args.skip:
            import umap
            emb = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                            random_state=SEED).fit_transform(Xs)
            scatter_embedding(emb, ys, "UMAP projection", out / "umap_visualization.png")
    write_run_info(out, args)


if __name__ == "__main__":
    main()
