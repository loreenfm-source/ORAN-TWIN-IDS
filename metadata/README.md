# Metadata

* **`features.md`** – the 25 raw signals (f0–f24), their statistical columns,
  units, protocol layers, variability, injection targets and ablation groups
  (mirrors paper Table 9).
* **`reservation_cluster_map.json`** – maps each of the 1,116 source
  reservations (`RESERVATION-<id>`) to its Colosseum cluster (`cluster_1`,
  `cluster_2`, `cluster_3`), taken from the source directory layout
  `cluster_*/slicing_*/scheduling_*/RESERVATION-*`. Stage 2 uses it to fill the
  `cluster` column of `windows_meta.csv`; `make_split_masks.py` and the LOCO
  experiment rely on that column.
* **`reservation_configuration.csv`** – the same 1,116 reservations with
  `cluster`, `slicing` (`slicing_1`–`slicing_5`) and `scheduling`
  (`scheduling_0`, `scheduling_2`), from the same directory layout.
  `make_split_masks.py` splits within each of the 30 configuration cells.

`windows_meta.csv` itself is part of the Zenodo release (not of this
repository). It has one row per window, row-aligned with `windows_scaled.npz`
and both CSV variants:

| Column | Meaning |
|---|---|
| `imsi` | UE identifier (the same eight IMSIs recur in every reservation) |
| `reservation_id` | source reservation (experimental session) |
| `cluster` | Colosseum cluster of the reservation |
| `start_timestamp` | timestamp of the first time step of the window (ISO 8601) |
| `label` | point-wise label of the first time step (0 everywhere; attack labels are in the CSVs) |

`imsi` is stored as an integer, so leading zeros are dropped: `1010123456002`
is the IMSI `001010123456002` (MCC 001, MNC 01, test network); the source UE
folders use the 15-digit form.
