#!/usr/bin/env python3
"""Sequence models on raw 50 x 25 windows (supplementary Section S2.7, Tables S8-S9).

1. Draw 500,000 benign windows from ``windows_scaled.npz`` within the masks,
   in the masks' train/test ratio (seed 42), each sorted; training rows first.
2. Inject stealth attacks with ``oran_twin_ids.injection`` (Table 4 stealth
   parameters, seed 42) - a separate injection from the released CSV.
3. Train a 2-layer bidirectional LSTM and a 4-block dilated causal TCN for the
   binary and the four-class task: Adam (lr 1e-3, weight decay 1e-4), cosine
   annealing, 30 epochs, batch 512, gradient clipping 1.0, class-weighted
   cross-entropy (weights N / (K * count), mean 1). Metrics are those of the
   final epoch; no checkpoint is selected on test data.
4. Prevalence analysis: binary models are trained again, independently, as in
   the paper (``--reuse_binary_models`` evaluates the step-3 models instead) and
   evaluated at 25/10/5/1 % test attack prevalence.

Requires PyTorch; a CUDA GPU is strongly recommended. Results vary slightly
between runs and devices because cuDNN kernels are not fully deterministic.

Example::

    python evaluation/temporal_dl_benchmark.py --windows data/windows_scaled.npz \
        --masks_dir data --output_dir results/temporal_dl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oran_twin_ids.cli import require_file, setup_logging, write_run_info  # noqa: E402
from oran_twin_ids.evaluation import SEED, downsample_attacks, precision_at_recall  # noqa: E402
from oran_twin_ids.features import CLASS_NAMES  # noqa: E402
from oran_twin_ids.injection import VARIANTS, inject_windows  # noqa: E402
from oran_twin_ids.splits import load_masks  # noqa: E402
from oran_twin_ids.variant import open_windows  # noqa: E402

log = logging.getLogger("temporal_dl")

N_SUBSAMPLE = 500_000
BATCH_SIZE = 512
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
PREVALENCE_LEVELS = (0.25, 0.10, 0.05, 0.01)
# multi-class label ids of the DL benchmark: 0 benign, 1 dos, 2 scan, 3 phy
CLASS_IDS = {name: i for i, name in enumerate(CLASS_NAMES)}


class LSTMClassifier(nn.Module):
    """2-layer bidirectional LSTM, mean-pooled over time, MLP head."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, n_layers: int = 2,
                 n_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if n_layers > 1 else 0.0)
        self.classifier = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(hidden_dim, n_classes))

    def forward(self, x):  # x: (batch, T, F)
        out, _ = self.lstm(x)
        return self.classifier(out.mean(dim=1))


class TCNBlock(nn.Module):
    """Residual block of two dilated causal convolutions with batch norm."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # left context; the future side is trimmed
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):  # x: (batch, channels, T)
        t = x.size(2)
        out = self.dropout(self.relu(self.norm1(self.conv1(x)[:, :, :t])))
        out = self.dropout(self.relu(self.norm2(self.conv2(out)[:, :, :t])))
        return self.relu(out + x)


class TCNClassifier(nn.Module):
    """1x1 input projection, dilations 1, 2, 4, 8, global average pooling, MLP head."""

    def __init__(self, input_dim: int, channels: int = 64, kernel_size: int = 3,
                 n_layers: int = 4, n_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, channels, 1)
        self.blocks = nn.ModuleList([TCNBlock(channels, kernel_size, 2 ** i, dropout)
                                     for i in range(n_layers)])
        self.classifier = nn.Sequential(nn.Linear(channels, channels // 2), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(channels // 2, n_classes))

    def forward(self, x):  # x: (batch, T, F)
        x = self.input_proj(x.permute(0, 2, 1))
        for block in self.blocks:
            x = block(x)
        return self.classifier(x.mean(dim=2))


MODELS = {"LSTM": LSTMClassifier, "TCN": TCNClassifier}


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_subsample(windows: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray,
                    n_total: int, seed: int):
    """Train/test window indices drawn within the masks (sorted, train first)."""
    rng = np.random.RandomState(seed)
    tr_pool, te_pool = np.flatnonzero(train_mask), np.flatnonzero(test_mask)
    n_tr = int(n_total * len(tr_pool) / (len(tr_pool) + len(te_pool)))  # the masks' ratio
    tr = np.sort(rng.choice(tr_pool, min(n_tr, len(tr_pool)), replace=False))
    te = np.sort(rng.choice(te_pool, min(n_total - n_tr, len(te_pool)), replace=False))
    return tr, te


def loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0,
                      pin_memory=torch.cuda.is_available())


def train(model: nn.Module, X: np.ndarray, y: np.ndarray, device, epochs: int,
          X_monitor=None, y_monitor=None) -> nn.Module:
    """Train with class-weighted cross-entropy; returns the final-epoch model.

    If monitoring data is given, its F1 is logged every 5 epochs for
    information only - it never influences training or model selection.
    """
    counts = np.bincount(y)
    weights = torch.tensor(len(y) / (len(counts) * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    data = loader(X, y, shuffle=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in data:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            total += loss.item()
        scheduler.step()
        if X_monitor is not None and (epoch % 5 == 0 or epoch == epochs):
            pred = predict(model, X_monitor, device).argmax(1)
            avg = "binary" if len(counts) == 2 else "weighted"
            log.info("    epoch %2d/%d loss %.4f | F1 %.4f", epoch, epochs, total / len(data),
                     f1_score(y_monitor, pred, average=avg, zero_division=0))
    return model


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, device) -> np.ndarray:
    """Softmax class probabilities."""
    model.eval()
    out = []
    for start in range(0, len(X), BATCH_SIZE):
        xb = torch.from_numpy(X[start:start + BATCH_SIZE]).to(device)
        out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="LSTM/TCN benchmark on raw windows (S2.7).")
    p.add_argument("--windows", type=Path, required=True, help="windows_scaled.npz")
    p.add_argument("--masks_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, default=Path("results/temporal_dl"))
    p.add_argument("--n_subsample", type=int, default=N_SUBSAMPLE)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--reuse_binary_models", action="store_true",
                   help="use the step-3 binary models for the prevalence analysis")
    args = p.parse_args(argv)
    setup_logging()
    require_file(p, args.windows, "--windows")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    windows, extracted = open_windows(args.windows, out)
    train_mask, test_mask = load_masks(args.masks_dir, n_rows=len(windows))
    tr, te = build_subsample(windows, train_mask, test_mask, args.n_subsample, args.seed)
    sub = np.concatenate([windows[tr], windows[te]]).astype(np.float32)
    del windows
    if extracted is not None:
        extracted.unlink()
    X, types = inject_windows(sub, VARIANTS["stealth"], args.seed)
    y_multi = np.array([CLASS_IDS[t] for t in types], dtype=np.int64)
    y_bin = (y_multi > 0).astype(np.int64)
    n_tr = len(tr)
    X_tr, X_te = X[:n_tr], X[n_tr:]
    log.info("Subsample: train %s | test %s | attack fraction %.3f",
             f"{n_tr:,}", f"{len(te):,}", y_bin.mean())
    n_features = X.shape[2]

    results, binary_models = [], {}
    for name, cls in MODELS.items():
        for task, y, n_classes, avg in (("binary", y_bin, 2, "binary"),
                                        ("multiclass", y_multi, 4, "weighted")):
            seed_everything(args.seed)
            model = cls(n_features, n_classes=n_classes).to(device)
            log.info("%s %s: %s parameters", name, task, f"{n_parameters(model):,}")
            train(model, X_tr, y[:n_tr], device, args.epochs, X_te, y[n_tr:])
            pred = predict(model, X_te, device).argmax(1)
            results.append({"model": name, "task": task, "parameters": n_parameters(model),
                            "f1": f1_score(y[n_tr:], pred, average=avg, zero_division=0),
                            "accuracy": accuracy_score(y[n_tr:], pred)})
            if task == "binary":
                binary_models[name] = model
    res = pd.DataFrame(results)
    res.to_csv(out / "temporal_dl_results.csv", index=False)
    print(res.round(4).to_string(index=False))

    pr_rows = []
    for name, cls in MODELS.items():
        if args.reuse_binary_models:
            model = binary_models[name]
        else:
            seed_everything(args.seed)
            model = train(cls(n_features, n_classes=2).to(device), X_tr, y_bin[:n_tr],
                          device, args.epochs)
        y_te = y_bin[n_tr:]
        scores = predict(model, X_te, device)[:, 1]
        for frac in PREVALENCE_LEVELS:
            idx = downsample_attacks(y_te, frac)
            y, s = y_te[idx], scores[idx]
            pr_rows.append({"model": name, "prevalence": f"{frac:.0%}",
                            "actual_prevalence": round(float(y.mean()), 4),
                            "pr_auc": average_precision_score(y, s),
                            "f1_at_0.5": f1_score(y, (s >= 0.5).astype(int), zero_division=0),
                            "prec_at_recall_0.90": precision_at_recall(y, s, 0.90),
                            "prec_at_recall_0.95": precision_at_recall(y, s, 0.95)})
    pr = pd.DataFrame(pr_rows)
    pr.to_csv(out / "temporal_dl_prevalence.csv", index=False)
    print(pr.round(4).to_string(index=False))
    (out / "subsample.json").write_text(json.dumps({"n_train": int(n_tr), "n_test": int(len(te)),
                                                    "seed": args.seed}), encoding="utf-8")
    write_run_info(out, args)


if __name__ == "__main__":
    main()
