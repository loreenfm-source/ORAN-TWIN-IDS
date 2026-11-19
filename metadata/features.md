# Feature reference

Each window of `windows_scaled.npz` has shape (50, 25): 50 time steps of the 25
raw telemetry signals below, in this order (f0 … f24). The tabular files
`real_strong.csv` and `real_stealth.csv` contain three statistics per signal,
computed over the 50 time steps of the window after attack injection:

* `f{i}_mean` – mean
* `f{i}_std` – population standard deviation (ddof = 0)
* `f{i}_p95` – 95th percentile (linear interpolation)

Column order: `f0_mean, f0_std, f0_p95, f1_mean, …, f24_p95, attack_type, label`
— 75 feature columns + 2 label columns = **77 columns**. All values are in the
StandardScaler-normalised space of Stage 2 (see the main README).

| Index | Source column | Description | Unit | Layer | Variability | Notes |
|---|---|---|---|---|---|---|
| f0 | `tx_brate downlink [Mbps]` | Downlink transmission bitrate | Mbps | MAC/RLC | Variable | DoS target; group A |
| f1 | `rx_brate uplink [Mbps]` | Uplink reception bitrate | Mbps | MAC/RLC | Variable | group A |
| f2 | `dl_mcs` | Downlink modulation and coding scheme | index | PHY | Variable | group C |
| f3 | `ul_mcs` | Uplink modulation and coding scheme | index | PHY | Variable | PHY target; group C |
| f4 | `dl_n_samples` | Number of downlink PHY samples | count | PHY | Variable | |
| f5 | `ul_n_samples` | Number of uplink PHY samples | count | PHY | Variable | |
| f6 | `dl_buffer [bytes]` | Downlink buffer occupancy | bytes | MAC | Variable | |
| f7 | `ul_buffer [bytes]` | Uplink buffer occupancy | bytes | MAC | Variable | |
| f8 | `tx_pkts downlink` | Transmitted downlink packets | packets | MAC | Variable | Scan target; group B |
| f9 | `rx_pkts uplink` | Received uplink packets | packets | MAC | Variable | group B |
| f10 | `tx_errors downlink (%)` | Downlink packet error rate | % | MAC | **Constant** | group B |
| f11 | `rx_errors uplink (%)` | Uplink packet error rate | % | MAC | Variable | group B |
| f12 | `ul_rssi` | Uplink received signal strength | dBm | PHY | **Constant** | |
| f13 | `ul_sinr` | Uplink signal-to-interference-plus-noise ratio | dB | PHY | Variable | PHY target; group C |
| f14 | `sum_requested_prbs` | Total requested physical resource blocks | PRBs | MAC | Variable | group A |
| f15 | `sum_granted_prbs` | Total granted physical resource blocks | PRBs | MAC | Variable | DoS target; group A |
| f16 | `slice_prb` | PRB allocation of the network slice | PRBs | RAN | Variable | group A |
| f17 | `dl_cqi` | Downlink channel quality indicator | index | PHY | Variable | group C |
| f18 | `phr` | Power headroom report | dB | PHY | Variable | |
| f19 | `mgen_flow_count` | Number of active MGEN flows | count | App | **Constant (zero)** | |
| f20 | `mgen_total_pkts` | Packets observed by MGEN | packets | App | **Constant (zero)** | |
| f21 | `mgen_total_payload` | Payload observed by MGEN | bytes | App | **Constant (zero)** | |
| f22 | `mgen_avg_pkt_size` | Average MGEN packet size | bytes | App | **Constant (zero)** | |
| f23 | `mgen_small_pkt_ratio` | Share of packets ≤ 100 bytes | ratio | App | **Constant (zero)** | |
| f24 | `mgen_missing` | MGEN record missing for the time step | binary | App | **Constant (zero)** | 1 before scaling |

Layer key: App = application (MGEN); RAN = RAN scheduler; MAC = medium access
control; PHY = physical layer.

**Effective dimensionality.** f10, f12 and f19–f24 are constant in every
window, so 17 signals (51 columns) carry information. Constant columns are 0
after scaling. The MGEN columns are constant because of how Stage 1 joins the
`mgen.csv` records (see "Known issues" in the main README), not because the
source files are empty: their timestamps are times of day without a date and
are never matched to the KPM time grid, and the first UE's file is used for
every UE of a reservation. 5,535 of the 6,881 per-UE MGEN logs contain packets.
The columns are kept for schema compatibility.

There is no f25: `PRB_util` (sum_granted_prbs / 50), described in v1, is not
part of the windows or the CSVs.

**Attack injection targets** (Stage 3): DoS raises f0 and f15, Scan raises f8,
PHY lowers f13 and f3. No other signal is modified.

**Ablation groups** (supplementary Table S6), used by
`evaluation/feature_group_ablation.py`:

| Group | Signals | Columns | Associated attack |
|---|---|---|---|
| A – bitrate / PRB | f0, f1, f14, f15, f16 | 15 | DoS |
| B – packets / errors | f8, f9, f10, f11 | 12 | Scan |
| C – radio quality | f2, f3, f13, f17 | 12 | PHY |

Removing all three groups leaves 36 columns (f4–f7, f12, f18–f24), of which 15
vary.

Plot aliases (SHAP figures, paper Table 10) are defined in
`oran_twin_ids/features.py` (`FEATURE_ALIASES`), e.g. `f8_std` → `tx_pkts_dl_std`.
