import zipfile

import numpy as np
import pandas as pd
import pytest
from conftest import data_dir

from oran_twin_ids.extraction import percentile_linear, window_statistics, write_feature_csv
from oran_twin_ids.features import CSV_COLUMNS
from oran_twin_ids.injection import VARIANTS, inject_windows


def loop_reference(windows):
    """Per-window loop of the original variant notebooks."""
    rows = []
    for w in windows:
        r = {}
        for f in range(w.shape[1]):
            col = w[:, f]
            r[f"f{f}_mean"], r[f"f{f}_std"] = col.mean(), col.std()
            r[f"f{f}_p95"] = np.percentile(col.astype(np.float64), 95)
        rows.append(r)
    return pd.DataFrame(rows)


def test_mean_std_bit_identical_to_loop():
    w = (np.random.RandomState(0).randn(300, 50, 25) * 3).astype(np.float32)
    ours, ref = window_statistics(w), loop_reference(w)
    for c in ours.columns:
        if c.endswith("_p95"):
            np.testing.assert_allclose(ours[c], ref[c], rtol=0, atol=1e-6)
        else:
            np.testing.assert_array_equal(ours[c].to_numpy(), ref[c].to_numpy(), err_msg=c)


@pytest.mark.parametrize("n", [1, 2, 7, 50, 51])
def test_percentile_matches_numpy(n):
    x = np.random.RandomState(n).randn(40, n).astype(np.float32)
    ours = percentile_linear(x)
    assert ours.dtype == np.float64
    np.testing.assert_allclose(ours, np.percentile(x.astype(np.float64), 95, axis=1),
                               rtol=0, atol=1e-6)


def test_csv_schema(tmp_path):
    w = np.random.RandomState(1).randn(250, 50, 25).astype(np.float32)
    types = np.array(["benign", "dos", "scan", "phy", "benign"] * 50, dtype=object)
    path = tmp_path / "x.csv"
    write_feature_csv(w, types, path, chunk=60)
    df = pd.read_csv(path)
    assert list(df.columns) == CSV_COLUMNS and len(df) == 250
    assert (df["label"] == (df["attack_type"] != "benign")).all()


@pytest.mark.data
def test_reproduces_released_rows_bytewise(tmp_path):
    """First 2,000 rows of both released CSVs regenerated from windows_scaled.npz."""
    d = data_dir()
    if d is None:
        pytest.skip("set ORAN_TWIN_IDS_DATA to the directory of the released files")
    n = 2000
    with zipfile.ZipFile(d / "windows_scaled.npz") as zf, zf.open("windows.npy") as f:
        version = np.lib.format.read_magic(f)
        reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                  else np.lib.format.read_array_header_2_0)
        reader(f)
        w =np.frombuffer(f.read(n * 50 * 25 * 4), dtype=np.float32).reshape(n, 50, 25)
    for variant in ("strong", "stealth"):
        injected, types = inject_windows(w, VARIANTS[variant], 42)
        out = tmp_path / f"{variant}.csv"
        write_feature_csv(injected, types, out, chunk=777)
        with open(d / f"real_{variant}.csv", encoding="utf-8") as fh:
            ref = [fh.readline() for _ in range(n + 1)]
        assert out.read_text(encoding="utf-8").splitlines(keepends=True) == ref
