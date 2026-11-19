from collections import Counter

import numpy as np
import pytest

from oran_twin_ids.injection import VARIANTS, inject_window, inject_windows

T, F = 50, 25
TARGETS = {"dos": {0, 15}, "scan": {8}, "phy": {3, 13}}


@pytest.mark.parametrize("variant", ["strong", "stealth"])
def test_only_target_segment_changes(variant):
    rng = np.random.RandomState(1)
    windows = rng.randn(400, T, F).astype(np.float32)
    original = windows.copy()
    out, types = inject_windows(windows, VARIANTS[variant], seed=42)
    np.testing.assert_array_equal(windows, original)  # input untouched
    max_len = int(T * VARIANTS[variant].duration[1])
    for w0, w1, kind in zip(original, out, types):
        diff = w1 - w0
        if kind == "benign":
            assert not diff.any()
            continue
        assert set(np.flatnonzero(np.abs(diff).sum(0))) <= TARGETS[kind]
        steps = np.flatnonzero(np.abs(diff).sum(1))
        assert steps.max() - steps.min() < max_len
        assert np.sign(diff.sum()) == (-1 if kind == "phy" else 1)


def test_deterministic():
    w = np.random.RandomState(2).randn(100, T, F).astype(np.float32)
    a, ta = inject_windows(w, VARIANTS["stealth"], seed=42)
    b, tb = inject_windows(w, VARIANTS["stealth"], seed=42)
    np.testing.assert_array_equal(a, b)
    assert list(ta) == list(tb)


def _notebook_dl_injection(windows, seed=42):
    """Reference copy of the injection of the original temporal DL notebook."""
    rng = np.random.RandomState(seed)
    X = windows.copy().astype(np.float32)
    y = np.zeros(len(X), dtype=np.int64)
    for i in range(len(X)):
        if rng.rand() >= 0.25:
            continue
        attack_id = rng.randint(1, 4)
        dur = int(T * rng.uniform(0.1, 0.35))
        start = rng.randint(0, T - dur)
        env = np.ones(dur, np.float32)
        k = max(1, dur // 3)
        env[:k] = np.linspace(0, 1, k)
        env[-k:] = np.linspace(1, 0, k)
        delta = np.zeros((dur, F), np.float32)
        if attack_id == 1:
            delta[:, 0] = rng.uniform(0.4, 0.9)
            delta[:, 15] = rng.uniform(0.3, 0.7)
        elif attack_id == 2:
            delta[:, 8] = rng.uniform(0.4, 0.9)
        else:
            delta[:, 13] = -rng.uniform(0.4, 0.9)
            delta[:, 3] = -rng.uniform(0.2, 0.5)
        X[i, start:start + dur] += env[:, None] * delta
        y[i] = attack_id
    return X, y


def test_shared_injection_equals_original_dl_notebook():
    w = np.random.RandomState(3).randn(3000, T, F).astype(np.float32)
    ref_X, ref_y = _notebook_dl_injection(w)
    X, types = inject_windows(w, VARIANTS["stealth"], seed=42)
    ids = {"benign": 0, "dos": 1, "scan": 2, "phy": 3}
    np.testing.assert_array_equal(X, ref_X)
    np.testing.assert_array_equal(np.array([ids[t] for t in types]), ref_y)


@pytest.mark.slow
@pytest.mark.parametrize("variant,expected", [
    ("strong", {"benign": 2_704_020, "dos": 300_226, "scan": 300_878, "phy": 300_997}),
    ("stealth", {"benign": 2_704_314, "dos": 299_960, "scan": 301_077, "phy": 300_770}),
])
def test_full_size_class_counts_match_paper_table5(variant, expected):
    # The labels depend only on the random stream, not on the window content.
    rng = np.random.RandomState(42)
    w = np.zeros((T, F), np.float32)
    counts = Counter(inject_window(w, rng, VARIANTS[variant]) for _ in range(3_606_121))
    assert dict(counts) == expected
