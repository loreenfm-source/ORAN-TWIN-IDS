"""Models, metrics and plotting helpers shared by the evaluation scripts."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve

log = logging.getLogger(__name__)

SEED = 42
PREVALENCES = (None, 0.10, 0.05, 0.01)  # None = original test set (~25 % attacks)


def baseline_models() -> dict:
    """The seven baselines of paper Table 6 (fresh instances).

    Settings are library defaults except where stated; every model that takes a
    seed uses ``random_state=42`` (KNN and Gaussian NB have none). XGBoost and
    LightGBM are the instances of ``boosting_models``.
    """
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=SEED),
        "DecisionTree": DecisionTreeClassifier(random_state=SEED),
        "KNN": KNeighborsClassifier(),
        "NaiveBayes": GaussianNB(),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED),
        **boosting_models(),
    }


def boosting_models() -> dict:
    """XGBoost and LightGBM, used by every experiment (fresh instances).

    Library defaults (XGBoost 100 trees, depth 6, learning rate 0.3, ``hist``;
    LightGBM 100 trees, 31 leaves, learning rate 0.1) with ``random_state=42``.
    The number of classes is inferred from the labels.
    """
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    return {
        "XGBoost": XGBClassifier(eval_metric="logloss", n_jobs=-1, random_state=SEED),
        "LightGBM": LGBMClassifier(verbose=-1, n_jobs=-1, random_state=SEED),
    }


def precision_at_recall(y_true: np.ndarray, scores: np.ndarray, min_recall: float) -> float:
    """Highest precision over all operating points with recall >= ``min_recall``."""
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ok = recall >= min_recall
    return float(precision[ok].max()) if ok.any() else float("nan")


def downsample_attacks(y_true: np.ndarray, target_fraction: float,
                       seed: int = SEED) -> np.ndarray:
    """Indices of a test subset whose attack prevalence is ``target_fraction``.

    All benign rows are kept; attack rows are sampled without replacement.
    Indices are returned in their original order.
    """
    rng = np.random.RandomState(seed)
    benign = np.flatnonzero(y_true == 0)
    attack = np.flatnonzero(y_true == 1)
    needed = int(len(benign) * target_fraction / (1 - target_fraction))
    if needed > len(attack):
        log.warning("target prevalence %.2f needs %d attacks, only %d available",
                    target_fraction, needed, len(attack))
        needed = len(attack)
    chosen = rng.choice(attack, needed, replace=False)
    return np.sort(np.concatenate([benign, chosen]))


def save_confusion_matrix(cm: np.ndarray, labels, title: str, path: Path) -> None:
    """Save an annotated confusion-matrix heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm, cmap="Blues", xticklabels=labels, yticklabels=labels)
    # Cell labels are drawn here: seaborn 0.12 with matplotlib >= 3.8 annotates only the first row.
    cmap, vmax = plt.get_cmap("Blues"), cm.max()
    for (i, j), v in np.ndenumerate(cm):
        dark = np.mean(cmap(v / vmax if vmax else 0.0)[:3]) < 0.5
        ax.text(j + 0.5, i + 0.5, f"{int(v):,}", ha="center", va="center", fontsize=10,
                color="white" if dark else "#262626")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
