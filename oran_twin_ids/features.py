"""Feature schema of ORAN-TWIN-IDS (paper Table 9).

The 25 raw telemetry signals f0..f24 are stored, in this order, as the last
axis of every window in ``windows_scaled.npz``. The tabular CSV variants hold
three statistics per signal (``f{i}_mean``, ``f{i}_std``, ``f{i}_p95``) followed
by ``attack_type`` and ``label`` (77 columns in total).
"""

from __future__ import annotations

# Source-dataset column names, in window feature order (f0..f24).
# 19 RAN KPIs from the per-UE ``<IMSI>_metrics.csv`` files ...
KPM_FEATURES: list[str] = [
    "tx_brate downlink [Mbps]",  # f0  - DoS target
    "rx_brate uplink [Mbps]",    # f1
    "dl_mcs",                    # f2
    "ul_mcs",                    # f3  - PHY target
    "dl_n_samples",              # f4
    "ul_n_samples",              # f5
    "dl_buffer [bytes]",         # f6
    "ul_buffer [bytes]",         # f7
    "tx_pkts downlink",          # f8  - Scan target
    "rx_pkts uplink",            # f9
    "tx_errors downlink (%)",    # f10 - constant in the source data
    "rx_errors uplink (%)",      # f11
    "ul_rssi",                   # f12 - constant in the source data
    "ul_sinr",                   # f13 - PHY target
    "sum_requested_prbs",        # f14
    "sum_granted_prbs",          # f15 - DoS target
    "slice_prb",                 # f16
    "dl_cqi",                    # f17
    "phr",                       # f18
]
# ... and 6 application-layer columns derived from ``mgen.csv`` (f19..f24).
MGEN_FEATURES: list[str] = [
    "mgen_flow_count",
    "mgen_total_pkts",
    "mgen_total_payload",
    "mgen_avg_pkt_size",
    "mgen_small_pkt_ratio",
    "mgen_missing",
]
RAW_FEATURES: list[str] = KPM_FEATURES + MGEN_FEATURES
N_FEATURES = len(RAW_FEATURES)  # 25

# Short human-readable aliases used in plots (SHAP figures, Table 10).
FEATURE_ALIASES: list[str] = [
    "tx_brate_dl", "rx_brate_ul", "dl_mcs", "ul_mcs",
    "dl_n_samples", "ul_n_samples", "dl_buffer", "ul_buffer",
    "tx_pkts_dl", "rx_pkts_ul", "tx_errors_dl", "rx_errors_ul",
    "ul_rssi", "ul_sinr", "sum_req_prbs", "sum_granted_prbs",
    "slice_prb", "dl_cqi", "phr",
    "mgen_flows", "mgen_pkts", "mgen_payload",
    "mgen_avg_pkt", "mgen_small_ratio", "mgen_missing",
]

# Signals that are constant across the source dataset (Table 9).
CONSTANT_FEATURES: tuple[int, ...] = (10, 12, 19, 20, 21, 22, 23, 24)

# Raw indices modified by attack injection.
INJECTION_TARGETS: dict[str, int] = {
    "tx_brate": 0,
    "ul_mcs": 3,
    "tx_pkts": 8,
    "ul_sinr": 13,
    "sum_granted_prbs": 15,
}

STATS: tuple[str, ...] = ("mean", "std", "p95")
FEATURE_COLUMNS: list[str] = [f"f{i}_{s}" for i in range(N_FEATURES) for s in STATS]
LABEL_COLUMNS: tuple[str, str] = ("attack_type", "label")
CSV_COLUMNS: list[str] = FEATURE_COLUMNS + list(LABEL_COLUMNS)  # 77 columns

ATTACK_TYPES: tuple[str, ...] = ("dos", "scan", "phy")
CLASS_NAMES: tuple[str, ...] = ("benign",) + ATTACK_TYPES

# Injection-aligned ablation groups (supplementary Section S2.4).
ABLATION_GROUPS: dict[str, tuple[int, ...]] = {
    "A": (0, 1, 14, 15, 16),  # bitrate / PRB  -> DoS   (15 columns)
    "B": (8, 9, 10, 11),      # packet / error -> Scan  (12 columns)
    "C": (2, 3, 13, 17),      # radio quality  -> PHY   (12 columns)
}


def raw_index(column: str) -> int:
    """Return the raw signal index of a statistical column, e.g. ``'f13_std' -> 13``."""
    prefix = column.split("_", 1)[0]
    if not (prefix.startswith("f") and prefix[1:].isdigit()):
        raise ValueError(f"not a feature column: {column!r}")
    return int(prefix[1:])


def columns_for(indices: "tuple[int, ...] | list[int]", columns: list[str]) -> list[str]:
    """Return the columns of ``columns`` that belong to the given raw indices."""
    wanted = set(indices)
    return [c for c in columns if c in FEATURE_COLUMNS and raw_index(c) in wanted]


def readable_name(column: str) -> str:
    """Map ``'f8_std'`` to its plot alias ``'tx_pkts_dl_std'``."""
    stat = column.split("_", 1)[1]
    return f"{FEATURE_ALIASES[raw_index(column)]}_{stat}"
