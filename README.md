# ORAN-TWIN-IDS

Code for generating and validating **ORAN-TWIN-IDS**, an intrusion-detection
dataset derived from the Open RAN Commercial Traffic Twinning Dataset
(Bonati et al., 2024). The source dataset contains only benign
hardware-in-the-loop O-RAN telemetry (LTE traces from commercial base stations
in Madrid replayed on the Colosseum emulator). ORAN-TWIN-IDS turns it into
labelled sliding windows by injecting parameterised perturbations consistent
with denial-of-service, scanning and physical-layer interference at two
intensity levels.

This repository accompanies the IEEE Data Descriptions article *"Descriptor:
An Intrusion Detection Dataset Derived from Open RAN Commercial Traffic
Twinning Data (ORAN-TWIN-IDS)"* by L. Mahmoud, M. Liyanage and
S. Gangopadhyay. The data are archived on Zenodo:
<https://doi.org/10.5281/zenodo.18980685>.

**Scope.** Two released variants, `real_strong` and `real_stealth`, each with
3,606,121 windows described by 75 statistical features (mean, standard
deviation and 95th percentile of 25 telemetry signals), the benign window
archive `windows_scaled.npz`, per-window metadata and the train/test split
masks. A TimeGAN generator was evaluated but is **not** part of the release
(`experimental/`). Attacks are injected, not captured: see
[Limitations](#limitations).

---

## Repository structure

```
ORAN-TWIN-IDS/
├── oran_twin_ids/                 shared code
│   ├── features.py                feature schema, injection targets, ablation groups
│   ├── injection.py               attack injection (paper Table 4)
│   ├── extraction.py              mean / std / p95 feature extraction
│   ├── splits.py                  reservation-level train/test split
│   ├── data.py                    loading of CSVs, metadata and masks
│   ├── evaluation.py              models and metrics shared by evaluation/
│   ├── variant.py                 Stage 3-4 driver
│   └── cli.py                     command-line helpers
├── scripts/                       dataset generation pipeline
│   ├── generate_pointwise_dataset.py      Stage 1
│   ├── generate_real_windows.py           Stage 2 (+ windows_meta.csv)
│   ├── make_split_masks.py                train/test masks
│   ├── generate_real_strong_dataset.py    Stages 3-4, strong variant
│   ├── generate_real_stealth_dataset.py   Stages 3-4, stealth variant
│   └── verify_release.py                  consistency checks of released files
├── evaluation/                    validation experiments of the paper
│   ├── baseline_classifiers.py    Table 6, Tables S2-S5, Figs. S6-S7
│   ├── feature_relevance.py       Fig. 2, Fig. 3, Figs. S2-S5
│   ├── feature_group_ablation.py  Table S6
│   ├── decision_stumps.py         Table S7
│   ├── shap_analysis.py           Figs. S8-S9, Table 10
│   ├── loco.py                    Table 7
│   ├── prevalence_pr_auc.py       Fig. 4, Table S9 (tabular models)
│   ├── split_checks.py            random-split spread, per-configuration F1
│   └── temporal_dl_benchmark.py   Tables S8-S9 (LSTM, TCN)
├── experimental/                  TimeGAN generator (evaluated, not released)
├── metadata/                      features.md, reservation_cluster_map.json,
│                                  reservation_configuration.csv
├── tests/                         pytest suite
├── requirements.txt
├── CHANGELOG.md
├── LICENSE                        MIT (code)
└── DATASET_LICENSE.md             CC BY-SA 4.0 (data)
```

---

## Data

### Released files (Zenodo)

| File | Content | Rows |
|---|---|---|
| `real_strong.csv` | 75 features + `attack_type` + `label`, strong injection | 3,606,121 |
| `real_stealth.csv` | same, stealth injection | 3,606,121 |
| `windows_scaled.npz` | key `windows`: benign windows before injection, float32 (N, 50, 25), scaled | 3,606,121 |
| `windows_meta.csv` | `imsi, reservation_id, cluster, start_timestamp, label` per window (label = point-wise label of the first time step, always 0; attack labels are in the CSVs) | 3,606,121 |
| `split_train_mask.npy`, `split_test_mask.npy` | boolean train/test masks, reservation-level, stratified by cluster × slicing × scheduling | 3,606,121 |

In `windows_meta.csv`, `imsi` is stored as an integer, so leading zeros are
dropped: `1010123456002` is the IMSI `001010123456002` (MCC 001, MNC 01, test
network); the source UE folders use the 15-digit form.

The record also holds `real_strong.zip` and `real_stealth.zip` (each the
corresponding CSV, deflate-compressed) and `SHA256SUMS.txt` with the SHA-256
of every file (`sha256sum -c SHA256SUMS.txt`).

All files are **row-aligned**: row *i* of each CSV, window *i* of the NPZ,
row *i* of the metadata and entry *i* of each mask describe the same window.

Class counts (attacks are injected with probability 0.25, the three types
with equal probability):

| Variant | benign | dos | scan | phy |
|---|---|---|---|---|
| real_strong | 2,704,020 | 300,226 | 300,878 | 300,997 |
| real_stealth | 2,704,314 | 299,960 | 301,077 | 300,770 |

CSV schema: `f0_mean, f0_std, f0_p95, …, f24_p95, attack_type, label`
(77 columns; `attack_type` ∈ {benign, dos, scan, phy}, `label` = 1 for any
attack). The signal of each index, units and layers are listed in
[`metadata/features.md`](metadata/features.md). Signals f10, f12 and f19–f24
are constant, so 17 signals (51 columns) carry information.

Download the files from the Zenodo record into a directory of your choice
(the examples below use `data/`), unzip the two CSVs into the same folder as
the other released files, and check them with

```bash
unzip real_strong.zip
unzip real_stealth.zip
python scripts/verify_release.py --data_dir data
```

Besides schema, row and class counts and row alignment, it checks the masks'
SHA-256, the train/test class counts of both CSVs, and that
`scripts/make_split_masks.py` with default settings regenerates both masks
byte for byte from `windows_meta.csv`.

### Source dataset

To regenerate everything from scratch you need the Open RAN Commercial
Traffic Twinning Dataset (<https://github.com/wineslab/open-ran-commercial-traffic-twinning-dataset>,
CC BY-SA 4.0), laid out as `cluster_{1,2,3}/slicing_{1..5}/scheduling_{0,2}/RESERVATION-*/`
with per-UE KPM files `bs/<IMSI>_metrics.csv` and `ue_*/mgen.csv`
(1,116 reservations).

---

## Installation

The pinned versions require Python 3.10-3.12 (numpy 1.26.4 has no wheels for
Python 3.13); the experiments used Python 3.11.7. The evaluation scripts and
tests also ran with newer versions (Python 3.13, numpy 2.5, pandas 3.0,
scikit-learn 1.9.1, xgboost 3.4, lightgbm 4.7, PyTorch 2.15); exact numbers
can differ slightly from the paper.

```bash
git clone https://github.com/loreenfm-source/ORAN-TWIN-IDS.git
cd ORAN-TWIN-IDS
pip install -r requirements.txt
```

PyTorch is needed only for `evaluation/temporal_dl_benchmark.py` and
`experimental/`. Install the build matching your CUDA version from
<https://pytorch.org/get-started/locally/>; the paper used 2.6.0 with CUDA 12.4:

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Scripts are run from the repository root; no package installation is needed.

**Hardware.** The full pipeline handles 90.4 M point-wise rows and 3.6 M
windows (about 18 GB as float32). Stage 2 keeps the point-wise features and keys in
memory (about 11 GB, roughly twice that at peak while the CSV chunks are
joined) and needs about 18 GB of free disk space; Stages 3–4
extract the NPZ to a temporary 18 GB file. Loading one CSV variant takes
about 2–3 GB of RAM. The sequence-model benchmark was run on an NVIDIA
RTX A5000.

---

## Quick start

```python
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

df = pd.read_csv("data/real_stealth.csv")
train = np.load("data/split_train_mask.npy")
test = np.load("data/split_test_mask.npy")
X = df.filter(regex=r"^f\d+_").to_numpy()      # 75 feature columns
y = df["label"].to_numpy()                      # 0 benign, 1 attack
scaler = StandardScaler().fit(X[train])
model = XGBClassifier(eval_metric="logloss", random_state=42)
model.fit(scaler.transform(X[train]), y[train])
print("test F1:", f1_score(y[test], model.predict(scaler.transform(X[test]))))
```

Use the released masks rather than a random split, because overlapping
windows of the same reservation would otherwise appear in both partitions.

---

## Reproducing the dataset

| Stage | Command | Output |
|---|---|---|
| 1. point-wise table | `python scripts/generate_pointwise_dataset.py --root_dir <source>/dataset-kpm --output_dir outputs/pointwise` | `pointwise_dataset.csv` (90,404,084 rows, 29 columns), `manifest_progress.json`, `bad_files.log` |
| 2. sliding windows | `python scripts/generate_real_windows.py --pointwise_csv outputs/pointwise/pointwise_dataset.csv --output_dir outputs/windows` | `windows_scaled.npz`, `windows_meta.csv`, `scaler.joblib`, `feature_names.json` |
| split masks | `python scripts/make_split_masks.py --meta outputs/windows/windows_meta.csv --output_dir outputs/windows` | `split_train_mask.npy`, `split_test_mask.npy` |
| 3–4. strong variant | `python scripts/generate_real_strong_dataset.py --input outputs/windows/windows_scaled.npz --output_dir outputs/real_strong` | `real_strong.csv` |
| 3–4. stealth variant | `python scripts/generate_real_stealth_dataset.py --input outputs/windows/windows_scaled.npz --output_dir outputs/real_stealth` | `real_stealth.csv` |

Options: Stage 1 `--resample_freq` (100ms), `--label_manifest`,
`--max_reservations` (quick test runs; the run is resumable). Stage 2 `--window` (50), `--stride` (25),
`--cluster_map` (default `metadata/reservation_cluster_map.json`), `--save_raw`.
Masks `--train_frac` (0.7). Variants `--seed` (42), `--keep_injected`.

### Pipeline details

1. **Point-wise table.** The source KPMs are sampled every 250 ms. Each UE's
   KPM series is resampled onto a regular 100 ms grid (`--resample_freq`,
   default `100ms` as released; `250ms` gives the native cadence) with linear
   interpolation of gaps up to five steps and forward/backward filling, so
   about 60 % of the grid points are interpolated between measured samples;
   MGEN statistics are left-joined. Output: the 25
   signals plus `reservation_id`, `imsi`, `timestamp` and `label` (all 0).
2. **Windows.** Rows are grouped by **(reservation_id, IMSI)** — the source
   reuses the same eight IMSIs in all reservations, so grouping by IMSI alone
   would join unrelated sessions — and sorted by time. Windows of
   T = 50 steps are taken every 25 steps (50 % overlap). With T = 50 steps on
   the 100 ms grid and a stride of 25 steps, each window spans 5 s and
   consecutive windows start 2.5 s apart; each window holds about 20 measured
   samples. This timescale is consistent with O-RAN closed-loop control, in
   which near-real-time RIC applications operate on loops of 10 ms to 1 s and
   non-real-time control on loops above 1 s (Polese et al., "Understanding
   O-RAN", IEEE Commun. Surveys Tuts., 2023). Groups are
   visited in reservation-identifier order, which defines the row order of
   every released file. A `StandardScaler` is fitted on all windows flattened
   to (N·T, 25) and applied to every window. 6,820 groups from 1,104
   reservations contain at least one window.
3. **Attack injection** (`oran_twin_ids/injection.py`). Windows are processed
   in row order with one `RandomState(42)`. With probability 0.25 a window is
   attacked; the type is drawn uniformly from {dos, scan, phy}, a segment of
   length ⌊T·U(d_lo, d_hi)⌋ is placed uniformly in the window, and the target
   signals are shifted by a uniformly drawn peak amplitude shaped by a
   trapezoidal envelope (linear rise over the first third, plateau, linear
   fall over the last third). Nothing outside the segment and no non-target
   signal is changed; windows are not blended with other windows.

   | Parameter (normalised units) | Strong | Stealth |
   |---|---|---|
   | segment duration (fraction of T) | [0.2, 0.6] | [0.1, 0.35] |
   | DoS: tx_brate (f0) | +[2.0, 4.0] | +[0.4, 0.9] |
   | DoS: sum_granted_prbs (f15) | +[1.0, 3.0] | +[0.3, 0.7] |
   | Scan: tx_pkts (f8) | +[2.0, 5.0] | +[0.4, 0.9] |
   | PHY: ul_sinr (f13) | −[2.0, 4.0] | −[0.4, 0.9] |
   | PHY: ul_mcs (f3) | −[1.0, 2.0] | −[0.2, 0.5] |

   The order of random draws is part of the dataset definition (see the module
   docstring); it reproduces the class counts above exactly.
4. **Feature extraction.** Mean, population standard deviation and 95th
   percentile per signal over the 50 steps, in the column order above.

---

## Running the experiments

All experiments use the reservation-level split masks, load the variants the
same way (pandas' default parser, float64), fit a fresh `StandardScaler` on
the training rows only and use `random_state=42` for every seeded model. Replace `data/` with the directory
holding the released files; every script also takes `--output_dir`
(default `results/<experiment>`) and writes CSV tables, figures and a
`run_info.json` with its arguments and library versions.

```bash
# Table 6, Tables S2-S5, confusion matrices (Figs. S6-S7)
python evaluation/baseline_classifiers.py --strong_csv data/real_strong.csv --stealth_csv data/real_stealth.csv --masks_dir data

# Feature relevance, heatmap, t-SNE and UMAP (Figs. 2, 3, S2-S5)
python evaluation/feature_relevance.py --dataset data/real_stealth.csv

# Feature-group ablation (Table S6)
python evaluation/feature_group_ablation.py --dataset data/real_stealth.csv --masks_dir data

# Decision stumps (Table S7)
python evaluation/decision_stumps.py --strong_csv data/real_strong.csv --stealth_csv data/real_stealth.csv --masks_dir data

# TreeSHAP (Figs. S8-S9, Table 10)
python evaluation/shap_analysis.py --stealth_csv data/real_stealth.csv --masks_dir data

# Leave-one-cluster-out (Table 7)
python evaluation/loco.py --dataset data/real_stealth.csv --meta data/windows_meta.csv --masks_dir data

# Reduced attack prevalence, tabular models (Fig. 4, Table S9)
python evaluation/prevalence_pr_auc.py --stealth_csv data/real_stealth.csv --strong_csv data/real_strong.csv --masks_dir data

# Split checks: random stratified splits and per-configuration F1
python evaluation/split_checks.py --dataset data/real_stealth.csv --meta data/windows_meta.csv --masks_dir data

# LSTM and TCN on raw windows (Tables S8-S9); PyTorch + GPU
python evaluation/temporal_dl_benchmark.py --windows data/windows_scaled.npz --masks_dir data
```

| Script | Main outputs |
|---|---|
| `baseline_classifiers.py` | `<variant>/binary_results.csv`, `multiclass_results.csv`, `multiclass_perclass_reports/`, `*_confusion_matrices/` |
| `feature_relevance.py` | `feature_label_correlation.*`, `feature_mutual_information.*`, `rf_feature_importance.*`, `feature_correlation_heatmap.png`, `tsne_visualization.png`, `umap_visualization.png` |
| `feature_group_ablation.py` | `feature_group_ablation.csv` |
| `decision_stumps.py` | `stumps_<variant>.csv` |
| `shap_analysis.py` | `shap_bar_*.png`, `shap_beeswarm_*.png`, `shap_ranking_*.csv` |
| `loco.py` | `loco_results.csv`, `loco_summary_{binary,multiclass}.csv` |
| `prevalence_pr_auc.py` | `prevalence_results.csv`, `pr_curves_*.png` |
| `split_checks.py` | `random_splits.csv`, `per_configuration.csv`, `split_checks.json` |
| `temporal_dl_benchmark.py` | `temporal_dl_results.csv`, `temporal_dl_prevalence.csv` |

### Approximate runtime and memory per script

Wall-clock times measured in the v2 result runs (from the first to the last
log line, so including data loading) on one workstation
(Intel Core i9-13900K, 24 cores, 128 GB RAM, NVIDIA RTX A5000), with the
default arguments shown above. Some CPU runs overlapped with the GPU run.
Peak RAM was not recorded.

| Script | Wall-clock | Peak RAM |
|---|---|---|
| `generate_pointwise_dataset.py` (Stage 1) | not measured | not measured |
| `generate_real_windows.py` (Stage 2) | not measured | not measured |
| `make_split_masks.py` | not measured | not measured |
| `generate_real_{strong,stealth}_dataset.py` (Stages 3–4) | not measured | not measured |
| `verify_release.py` | not measured | not measured |
| `baseline_classifiers.py` (both variants) | 1 h 37 min | not measured |
| `feature_relevance.py` (Pearson, MI, Random Forest; `--skip heatmap tsne umap`) | 24 min | not measured |
| `feature_group_ablation.py` | 7 min | not measured |
| `decision_stumps.py` | 1 min | not measured |
| `shap_analysis.py` | 1 min | not measured |
| `loco.py` | 4 min | not measured |
| `prevalence_pr_auc.py` | 1 min | not measured |
| `split_checks.py` | 4 min | not measured |
| `temporal_dl_benchmark.py` (GPU) | 28–30 min per seed | not measured |

`temporal_dl_benchmark.py` and Stages 3–4 extract the NPZ to an ~18 GB
temporary file in `--output_dir`.

Protocol notes:

* **Baselines:** LogisticRegression(max_iter=1000), DecisionTree, KNN (k = 5),
  GaussianNB, RandomForest(200 trees), XGBoost(eval_metric="logloss"),
  LightGBM — library defaults otherwise, `random_state=42` where applicable.
  Multi-class metrics are weighted averages.
* **Feature relevance:** MI and the Random Forest importance are computed on
  all windows (descriptive, not a generalisation estimate); t-SNE/UMAP use a
  random 10,000-window sample. `--skip mi rf` avoids the slowest parts.
* **Stumps:** 50,000 training and 20,000 test windows sampled within the masks.
* **SHAP:** binary models on the training partition; 5,000 test windows with
  the test attack ratio.
* **LOCO:** clusters from `windows_meta.csv`; the TRTR reference is recomputed
  with the same model settings (100 trees) in the same run.
* **Prevalence:** models are trained once; only test attack windows are
  removed (seed 42). Precision at recall r is the highest precision among
  operating points with recall ≥ r.
* **Sequence models:** 500,000 windows drawn within the masks in their
  train/test ratio (349,778 / 150,222 for the released masks), a separate
  stealth injection with seed 42, final-epoch
  metrics (no selection on test data), class-weighted loss with weights of mean
  1. By default the binary models for the prevalence analysis are retrained
  independently, as in the paper (`--reuse_binary_models` reuses the first ones).
  2-layer bidirectional LSTM: 587,138 parameters; TCN: 103,650 parameters.

---

## Reproducibility

* **Seeds:** 42 for injection, sampling, splits of subsamples and all
  stochastic models. `verify_release.py --seed` only affects which rows are
  spot-checked.
* **Split:** within each (cluster, slicing, scheduling) configuration cell,
  the first 70 % of reservations in identifier order form the training set
  (`metadata/reservation_configuration.csv`; all 30 cells have 10 or more
  reservations). For the released data this gives 2,522,688 training and
  1,083,433 test windows (70.0 % / 30.0 %; 772 / 332 reservations), with no
  reservation in both partitions; overlapping windows therefore never straddle
  the boundary. Cluster shares: train 24.3 / 39.7 / 36.0 %, test
  25.4 / 38.3 / 36.3 % (clusters 1 / 2 / 3); every slicing/scheduling
  configuration has between 4.7 % and 17.4 % of the windows on each side.
* **Determinism:** Stages 3–4 reproduce the released CSVs byte for byte from
  `windows_scaled.npz` (checked on the first 2,000 rows of both variants and by
  the class counts). Feature extraction and scaling deliberately follow the
  floating-point behaviour of numpy 1.26 / scikit-learn 1.4, so newer library
  versions give identical files. Tree ensembles with multiple threads and
  cuDNN-based sequence models can differ slightly between runs and machines.
* **Library versions:** `requirements.txt` lists the environment used for the
  experiments (Python 3.11.7). The Stage 2 scaler of the release was fitted
  with scikit-learn 1.4.2.

Run the tests with `pip install pytest` and `python -m pytest`
(`-m "not slow"` skips the two full-size class-count simulations; tests marked
`data` additionally need `ORAN_TWIN_IDS_DATA` pointing to the released files).

---

## Limitations

* The source telemetry comes from the Colosseum emulator replaying LTE traces,
  not from a live production network.
* Attacks are rule-based perturbations of five signals in the normalised
  feature space; side effects of real attacks (buffer growth, retransmissions,
  BLER/CQI changes) are not modelled, and perturbed integer-valued signals can
  take values between valid levels. Validate detectors on real or emulated
  attack traffic before deployment.
* The 25 % attack fraction is a benchmarking choice, not an operational
  prevalence; `evaluation/prevalence_pr_auc.py` shows how to evaluate lower
  prevalences.
* **100 ms grid.** The source KPMs are sampled every 250 ms; the release uses
  a 100 ms grid with linear interpolation, so about 60 % of the grid points
  are interpolated and each window (5 s, 2.5 s stride) holds about 20
  measured samples. Stage 1 with `--resample_freq 250ms` gives the native
  cadence; the released data use the default `100ms`.
* The Stage 2 scaler is fitted on all windows (before splitting). It defines
  the space in which attacks are injected; the evaluation scripts refit a
  scaler on the training rows of the tabular features.
* Only three attack families are covered; O-RAN-specific threats such as xApp
  poisoning, KPM falsification or slice-isolation attacks are not.

### Known issues

* **MGEN features are constant.** `mgen.csv` timestamps are times of day
  without a date. Stage 1 resolves them to the current date, so they never
  match the 2024 KPM index and every MGEN column is zero-filled
  (`mgen_missing` = 1 before scaling); in addition, the first UE's `mgen.csv`
  of a reservation is used for all its UEs. The source logs are not empty:
  5,535 of the 6,881 per-UE `mgen.csv` files that match a KPM file contain
  packets (the rest hold only a header). Their time of day is on the same UTC
  clock as the KPM timestamps, and a join that takes the date from the KPM
  session (with a day rollover for the 102 sessions that cross midnight)
  tracks the UE's KPM downlink traffic. The released behaviour is kept so that
  the release can be reproduced; a corrected join is planned for a future
  dataset version. It will require a decision on the injection, which
  modifies KPM downlink signals but not the MGEN statistics that track them.
  The released features therefore describe RAN-level KPM telemetry only.
* f10 (downlink error rate) and f12 (uplink RSSI) are constant in the source
  telemetry.

---

## Changes in v2

v2 fixes the window grouping (reservation and IMSI), replaces positional
splits by reservation-level masks stratified by cluster, slicing and
scheduling, removes the window blending that leaked
labels through non-target features, removes the non-existent `PRB_util`
feature, drops the synthetic variants, adds `windows_meta.csv` and the masks
to the release, and replaces the notebooks by scripts. See
[CHANGELOG.md](CHANGELOG.md).

---

## Citation

If you use ORAN-TWIN-IDS, please cite the dataset
(<https://doi.org/10.5281/zenodo.18980685>) and the descriptor article
(L. Mahmoud, M. Liyanage, S. Gangopadhyay, *Descriptor: An Intrusion
Detection Dataset Derived from Open RAN Commercial Traffic Twinning Data
(ORAN-TWIN-IDS)*, IEEE Data Descriptions; bibliographic details will be added
on publication), as well as the source dataset:

> L. Bonati, R. Shirkhani, C. Fiandrino, S. Maxenti, S. D'Oro, M. Polese and
> T. Melodia, "Twinning commercial network traces on experimental Open RAN
> platforms," in *Proc. 30th Annual International Conference on Mobile
> Computing and Networking (MobiCom)*, 2024, pp. 1914–1921.

## Licence

Code: MIT License ([LICENSE](LICENSE)). Data: CC BY-SA 4.0
([DATASET_LICENSE.md](DATASET_LICENSE.md)), following the source dataset.

## Contact

Please report problems through the GitHub issue tracker of this repository.
