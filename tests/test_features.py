from oran_twin_ids import features as F


def test_schema_sizes():
    assert F.N_FEATURES == 25 and len(F.RAW_FEATURES) == 25 and len(F.FEATURE_ALIASES) == 25
    assert len(F.FEATURE_COLUMNS) == 75 and len(F.CSV_COLUMNS) == 77
    assert F.FEATURE_COLUMNS[:4] == ["f0_mean", "f0_std", "f0_p95", "f1_mean"]
    assert F.CSV_COLUMNS[-3:] == ["f24_p95", "attack_type", "label"]
    assert "PRB_util" not in F.RAW_FEATURES


def test_injection_targets_match_raw_names():
    names = {i: F.RAW_FEATURES[i] for i in F.INJECTION_TARGETS.values()}
    assert names == {0: "tx_brate downlink [Mbps]", 3: "ul_mcs", 8: "tx_pkts downlink",
                     13: "ul_sinr", 15: "sum_granted_prbs"}


def test_ablation_group_sizes():
    sizes = {g: len(F.columns_for(idx, F.FEATURE_COLUMNS)) for g, idx in F.ABLATION_GROUPS.items()}
    assert sizes == {"A": 15, "B": 12, "C": 12}
    removed = F.columns_for([i for g in F.ABLATION_GROUPS.values() for i in g], F.FEATURE_COLUMNS)
    remaining = [c for c in F.FEATURE_COLUMNS if c not in removed]
    assert len(remaining) == 36
    variable = [c for c in remaining if F.raw_index(c) not in F.CONSTANT_FEATURES]
    assert len(variable) == 15  # supplementary Section S2.4


def test_readable_names_are_unique():
    names = [F.readable_name(c) for c in F.FEATURE_COLUMNS]
    assert len(set(names)) == 75
    assert F.readable_name("f8_std") == "tx_pkts_dl_std"
