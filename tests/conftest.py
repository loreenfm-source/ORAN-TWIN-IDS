import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oran_twin_ids.features import KPM_FEATURES, MGEN_FEATURES

REPO_ROOT = Path(__file__).resolve().parents[1]

# (reservation, IMSI) -> number of 100 ms rows. The same IMSIs occur in every
# reservation, as in the source dataset; one group is shorter than a window.
GROUP_LENGTHS = {
    ("RESERVATION-100002", 1010123456002): 130,
    ("RESERVATION-100002", 1010123456003): 49,
    ("RESERVATION-100001", 1010123456002): 100,
    ("RESERVATION-100001", 1010123456003): 75,
    ("RESERVATION-100003", 1010123456002): 50,
    ("RESERVATION-100003", 1010123456003): 76,
}


def data_dir():
    """Directory with the released files, if ORAN_TWIN_IDS_DATA is set."""
    d = os.environ.get("ORAN_TWIN_IDS_DATA")
    return Path(d) if d else None


@pytest.fixture
def pointwise_csv(tmp_path):
    """Tiny point-wise table with shuffled rows."""
    rng = np.random.RandomState(0)
    frames = []
    for (res, imsi), n in GROUP_LENGTHS.items():
        ts = pd.date_range("2024-02-20 21:00:00", periods=n, freq="100ms")
        df = pd.DataFrame(rng.randn(n, len(KPM_FEATURES)), columns=KPM_FEATURES)
        for c in MGEN_FEATURES:
            df[c] = 0.0
        df["mgen_missing"] = 1.0
        df["reservation_id"], df["imsi"] = res, imsi
        df["timestamp"] = [t.isoformat() for t in ts]
        df["label"] = 0
        frames.append(df)
    df = pd.concat(frames, ignore_index=True).sample(frac=1.0, random_state=1)
    path = tmp_path / "pointwise_dataset.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def cluster_map(tmp_path):
    m = {"RESERVATION-100001": "cluster_1", "RESERVATION-100002": "cluster_1",
         "RESERVATION-100003": "cluster_2"}
    path = tmp_path / "cluster_map.json"
    path.write_text(json.dumps(m))
    write_configuration(m, tmp_path / "configuration.csv")
    return path


def write_configuration(cluster_map: dict, path):
    """Configuration table for ``make_split_masks.py`` (one slicing/scheduling cell per cluster)."""
    pd.DataFrame({"reservation_id": list(cluster_map), "cluster": list(cluster_map.values()),
                  "slicing": "slicing_1", "scheduling": "scheduling_0"}).to_csv(path, index=False)
    return path
