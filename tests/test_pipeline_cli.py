"""End-to-end runs of the documented command lines on small synthetic data."""

import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from conftest import GROUP_LENGTHS, REPO_ROOT, write_configuration

from oran_twin_ids.features import CSV_COLUMNS, RAW_FEATURES

WINDOW, STRIDE = 50, 25


def run(script, *args):
    cmd = [sys.executable, str(REPO_ROOT / script), *map(str, args)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert proc.returncode == 0, f"{script} failed:\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout


def test_stage2_windows_meta_alignment(tmp_path, pointwise_csv, cluster_map):
    out = tmp_path / "windows"
    run("scripts/generate_real_windows.py", "--pointwise_csv", pointwise_csv,
        "--output_dir", out, "--cluster_map", cluster_map, "--save_raw")
    meta = pd.read_csv(out / "windows_meta.csv")
    assert list(meta.columns) == ["imsi", "reservation_id", "cluster", "start_timestamp", "label"]
    raw = np.load(out / "windows_raw.npz")["windows"]
    scaled = np.load(out / "windows_scaled.npz")["windows"]
    assert raw.shape == scaled.shape == (len(meta), WINDOW, 25) and scaled.dtype == np.float32

    # expected windows: sorted (reservation, IMSI) groups, stride 25, short groups skipped
    expected = []
    for key in sorted(GROUP_LENGTHS):
        n = GROUP_LENGTHS[key]
        expected += [key] * (len(range(0, n - WINDOW + 1, STRIDE)) if n >= WINDOW else 0)
    assert list(zip(meta.reservation_id, meta.imsi)) == expected
    assert meta.cluster.tolist() == ["cluster_2" if r == "RESERVATION-100003" else "cluster_1"
                                     for r, _ in expected]

    pw = pd.read_csv(pointwise_csv)
    pw["ts"] = pd.to_datetime(pw.timestamp, format="ISO8601")
    for k, row in meta.iterrows():
        g = pw[(pw.reservation_id == row.reservation_id) & (pw.imsi == row.imsi)].sort_values("ts")
        start = int(np.flatnonzero(g.timestamp.to_numpy() == row.start_timestamp)[0])
        assert start % STRIDE == 0
        np.testing.assert_array_equal(raw[k], g[RAW_FEATURES].to_numpy(np.float32)[start:start + WINDOW])
    flat = scaled.reshape(-1, 25)
    variable = flat.std(0) > 0
    np.testing.assert_allclose(flat.mean(0), 0, atol=1e-5)
    np.testing.assert_allclose(flat.std(0)[variable], 1, atol=1e-4)

    run("scripts/make_split_masks.py", "--meta", out / "windows_meta.csv", "--output_dir", out,
        "--train_frac", "0.5", "--cluster_map", cluster_map,
        "--configuration", cluster_map.parent / "configuration.csv")
    train = np.load(out / "split_train_mask.npy")
    assert not set(meta.reservation_id[train]) & set(meta.reservation_id[~train])


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    """A small synthetic 'release': windows, both variants, meta and masks."""
    d = tmp_path_factory.mktemp("release")
    rng = np.random.RandomState(0)
    n_res, per_res = 12, 60
    windows = rng.randn(n_res * per_res, WINDOW, 25).astype(np.float32)
    windows[:, :, [10, 12, 19, 20, 21, 22, 23, 24]] = 0.0
    np.savez_compressed(d / "windows_scaled.npz", windows=windows)
    res = np.repeat([f"RESERVATION-1{i:05d}" for i in range(n_res)], per_res)
    meta = pd.DataFrame({"imsi": "1010123456002", "reservation_id": res,
                         "start_timestamp": "2024-02-20T21:00:00", "label": 0})
    meta.to_csv(d / "meta_in.csv", index=False)
    cmap = {r: f"cluster_{1 + int(r[-5:]) % 3}" for r in np.unique(res)}
    (d / "cmap.json").write_text(json.dumps(cmap))
    write_configuration(cmap, d / "configuration.csv")
    run("scripts/make_split_masks.py", "--meta", d / "meta_in.csv", "--output_dir", d,
        "--cluster_map", d / "cmap.json", "--configuration", d / "configuration.csv",
        "--write_meta")
    for v in ("strong", "stealth"):
        run(f"scripts/generate_real_{v}_dataset.py", "--input", d / "windows_scaled.npz",
            "--output_dir", d / v)
        (d / v / f"real_{v}.csv").replace(d / f"real_{v}.csv")
    return d


def test_variant_outputs(release):
    for v in ("strong", "stealth"):
        df = pd.read_csv(release / f"real_{v}.csv")
        assert list(df.columns) == CSV_COLUMNS and len(df) == 720
        assert set(df.attack_type) == {"benign", "dos", "scan", "phy"}
    assert not list(release.glob("*/_windows_scaled.npy"))  # temporary file removed


def test_evaluation_scripts_run(release, tmp_path):
    d = release
    common = ["--masks_dir", d]
    run("evaluation/baseline_classifiers.py", "--stealth_csv", d / "real_stealth.csv", *common,
        "--output_dir", tmp_path / "b", "--models", "DecisionTree", "NaiveBayes", "XGBoost")
    res = pd.read_csv(tmp_path / "b" / "real_stealth" / "multiclass_results.csv")
    assert list(res.model) == ["DecisionTree", "NaiveBayes", "XGBoost"]
    assert (tmp_path / "b" / "real_stealth" / "binary_confusion_matrices" / "XGBoost.png").is_file()
    run("evaluation/feature_group_ablation.py", "--dataset", d / "real_stealth.csv", *common,
        "--output_dir", tmp_path / "a")
    abl = pd.read_csv(tmp_path / "a" / "feature_group_ablation.csv")
    assert sorted(abl.n_features.unique()) == [36, 48, 51, 60, 63, 75]
    run("evaluation/decision_stumps.py", "--strong_csv", d / "real_strong.csv", *common,
        "--output_dir", tmp_path / "s")
    run("evaluation/prevalence_pr_auc.py", "--stealth_csv", d / "real_stealth.csv", *common,
        "--output_dir", tmp_path / "p")
    run("evaluation/loco.py", "--dataset", d / "real_stealth.csv", "--meta", d / "windows_meta.csv",
        *common, "--output_dir", tmp_path / "l")
    assert len(pd.read_csv(tmp_path / "l" / "loco_results.csv")) == 16
    run("evaluation/shap_analysis.py", "--stealth_csv", d / "real_stealth.csv", *common,
        "--output_dir", tmp_path / "sh")
    run("evaluation/feature_relevance.py", "--dataset", d / "real_stealth.csv",
        "--output_dir", tmp_path / "r")


def test_verify_release_runs(release):
    out = subprocess.run([sys.executable, str(REPO_ROOT / "scripts/verify_release.py"),
                          "--data_dir", release], capture_output=True, text=True, cwd=REPO_ROOT)
    # The synthetic release has the wrong size and class counts, so some checks fail
    # by design; schema and NPZ/CSV row alignment must pass.
    assert "[PASS] real_stealth: 77-column schema" in out.stdout
    assert "[PASS] real_stealth: NPZ/CSV alignment" in out.stdout
    assert "[PASS] masks disjoint and complementary" in out.stdout


def test_verify_release_fails_on_missing_csv(release, tmp_path):
    for f in release.iterdir():
        if f.is_file() and f.name != "real_strong.csv":
            shutil.copy(f, tmp_path / f.name)
    out = subprocess.run([sys.executable, str(REPO_ROOT / "scripts/verify_release.py"),
                          "--data_dir", tmp_path], capture_output=True, text=True, cwd=REPO_ROOT)
    assert f"[FAIL] real_strong.csv not found in {tmp_path} (unzip real_strong.zip first)" \
        in out.stdout
    assert "all checks passed" not in out.stdout
    assert out.returncode == 1
    assert "[PASS] real_stealth: 77-column schema" in out.stdout


def test_temporal_dl_parameter_counts_and_smoke(release, tmp_path):
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(REPO_ROOT / "evaluation"))
    from temporal_dl_benchmark import LSTMClassifier, TCNClassifier, n_parameters
    assert n_parameters(LSTMClassifier(25, n_classes=2)) == 587_138  # paper S2.7
    assert n_parameters(TCNClassifier(25, n_classes=2)) == 103_650
    del torch
    run("evaluation/temporal_dl_benchmark.py", "--windows", release / "windows_scaled.npz",
        "--masks_dir", release, "--output_dir", tmp_path / "dl", "--n_subsample", "400",
        "--epochs", "1")
    assert len(pd.read_csv(tmp_path / "dl" / "temporal_dl_prevalence.csv")) == 8
