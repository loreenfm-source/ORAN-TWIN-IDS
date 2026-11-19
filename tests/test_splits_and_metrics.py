import numpy as np
import pandas as pd
import pytest

from oran_twin_ids.evaluation import baseline_models, downsample_attacks, precision_at_recall
from oran_twin_ids.splits import check_disjoint, load_masks, reservation_split


def make_meta():
    rows = []
    for prefix, cl, n_res in (("1", "cluster_1", 10), ("2", "cluster_2", 7)):
        for r in range(n_res):
            rows += [{"reservation_id": f"RESERVATION-{prefix}{r:05d}", "cluster": cl,
                      "slicing": f"slicing_{1 + r % 2}", "scheduling": "scheduling_0"}] * (r + 3)
    return pd.DataFrame(rows).sample(frac=1.0, random_state=0).reset_index(drop=True)


def test_stratified_split_per_configuration_cell():
    meta = make_meta()
    train = reservation_split(meta, 0.7)
    check_disjoint(meta.reservation_id.to_numpy(), train)
    for _, cell in meta.groupby(["cluster", "slicing", "scheduling"]):
        ids = sorted(cell["reservation_id"].unique())
        got = sorted(cell.loc[train[cell.index], "reservation_id"].unique())
        assert got == ids[:int(round(0.7 * len(ids)))]  # first 70 % of each cell


def test_reservation_level_and_per_cluster():
    meta = make_meta()
    train = reservation_split(meta, 0.7, strata=("cluster",))
    check_disjoint(meta.reservation_id.to_numpy(), train)
    for cl, n_res in (("cluster_1", 10), ("cluster_2", 7)):
        ids = sorted(meta.loc[meta.cluster == cl, "reservation_id"].unique())
        got = sorted(meta.loc[(meta.cluster == cl) & train, "reservation_id"].unique())
        assert got == ids[:int(round(0.7 * n_res))]  # first 70 % in identifier order


def test_split_independent_of_row_order():
    meta = make_meta()
    train = reservation_split(meta)
    perm = np.random.RandomState(1).permutation(len(meta))
    np.testing.assert_array_equal(reservation_split(meta.iloc[perm].reset_index(drop=True)),
                                  train[perm])


def test_rejects_missing_clusters():
    meta = make_meta()
    meta.loc[0, "cluster"] = None
    with pytest.raises(ValueError):
        reservation_split(meta)
    with pytest.raises(ValueError):
        reservation_split(make_meta().drop(columns="slicing"))


def test_load_masks_validation(tmp_path):
    np.save(tmp_path / "split_train_mask.npy", np.array([True, False, True]))
    np.save(tmp_path / "split_test_mask.npy", np.array([False, True, True]))
    with pytest.raises(ValueError, match="overlap"):
        load_masks(tmp_path)
    np.save(tmp_path / "split_test_mask.npy", np.array([False, True, False]))
    with pytest.raises(ValueError, match="rows"):
        load_masks(tmp_path, n_rows=4)
    train, test = load_masks(tmp_path, n_rows=3)
    assert train.sum() == 2 and test.sum() == 1


@pytest.mark.parametrize("frac", [0.10, 0.05, 0.01])
def test_downsample_keeps_benign_and_hits_prevalence(frac):
    y = np.array([0] * 7500 + [1] * 2500)
    idx = downsample_attacks(y, frac)
    assert (y[idx] == 0).sum() == 7500
    assert abs(y[idx].mean() - frac) < 1e-3
    assert np.all(np.diff(idx) > 0)


def test_precision_at_recall():
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])
    assert precision_at_recall(y, s, 0.66) == 1.0
    assert precision_at_recall(y, s, 1.0) == pytest.approx(0.75)


def test_seven_baselines():
    assert list(baseline_models()) == ["LogisticRegression", "DecisionTree", "KNN", "NaiveBayes",
                                       "RandomForest", "XGBoost", "LightGBM"]


def test_every_seeded_model_uses_seed_42():
    for name, model in baseline_models().items():
        params = model.get_params()
        if "random_state" in params:
            assert params["random_state"] == 42, name
