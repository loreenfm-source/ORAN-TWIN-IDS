# Changelog

## v2.0.0

Release accompanying the revised descriptor article. The data must be
regenerated with this version; v1 files are not compatible.

### Fixed
- **Window grouping.** Stage 2 grouped rows by IMSI only, but the source
  dataset reuses eight IMSIs across all reservations, so most v1 windows mixed
  unrelated sessions. Rows are now grouped by (reservation_id, IMSI) and sorted
  by time; the dataset has 3,606,121 windows (v1: 3,616,152).
- **Label leakage through blending.** v1 blended every attacked window with a
  random benign window, which created non-integer values in all signals and
  revealed the label through never-targeted features. Injection now perturbs
  only the target signals inside the attack segment, with a trapezoidal ramp
  envelope; stealth magnitudes were recalibrated (paper Table 4).
- **Train/test split.** The positional 70/30 split (with a 2-row gap) placed the
  whole test set in one cluster. All experiments now use reservation-level
  masks: the first 70 % of reservations (identifier order) of each
  (cluster, slicing, scheduling) cell. A per-cluster split would put 73 % of
  the test windows in slicing_5 and no scheduling_2 reservation of slicings
  1–4 in the test set, because reservation identifiers follow the
  experiment schedule.
- **One evaluation path.** Every experiment loads the variants with
  `load_variant` (pandas defaults), scales with a StandardScaler fitted on the
  training rows (`scale_partition`) and seeds every model with 42; LOCO uses
  the same XGBoost/LightGBM instances as the other experiments. The paper's
  notebooks differed in these details; before the re-run, the evaluation
  scripts were validated by reproducing the earlier (v1-mask) results exactly.
- **Metadata alignment.** `windows_meta.csv` is written in the same loop as the
  windows, so it is row-aligned with the NPZ and CSVs by construction, and it
  now includes `reservation_id` and `cluster`.
- PHY injection in the sequence-model benchmark perturbs `ul_mcs` (f3), as in
  the released variants (v1 used `dl_mcs`).
- Sequence-model results are taken from the final epoch (v1 selected the best
  epoch on the test set); class weights are normalised to mean 1.
- Decision stumps take column names from the CSV header; LOCO computes its own
  TRTR reference; confusion matrices come from the models behind the tables.
- TimeGAN fidelity is measured against a random sample of all windows instead
  of the first 20,000 rows.

### Changed
- 25 raw signals (f0–f24), 75 statistical columns, 77-column CSVs. The
  `PRB_util` column (f25 in v1 documentation) is no longer written by Stage 1
  and was never part of the v2 windows.
- Stage 1: fixed 29-column output schema, `--label_manifest` and
  `--max_reservations` options, vectorised row assembly (output unchanged).
- Stage 1: grid inference replaced by an explicit `--resample_freq` option
  (default 100ms, as released; 250ms gives the native cadence).
- `verify_release.py` reads the NPZ header with the public numpy API (numpy
  1.x and 2.x) and reports a missing variant CSV as a failure.
- Stage 2: disk-backed window assembly, compact keys, no `windows_raw.npz`
  unless `--save_raw`; scaling reproduces the released archive bit for bit.
- Stages 3–4: one shared implementation (`oran_twin_ids/`), vectorised feature
  extraction that reproduces the released CSVs byte for byte.
- All paths are command-line arguments; personal paths were removed.
- Dependencies pinned to the experiment environment; PyTorch installed
  separately.

### Added
- `scripts/make_split_masks.py`, `scripts/verify_release.py`.
- `evaluation/` scripts replacing the notebooks (baselines, feature relevance,
  feature-group ablation, decision stumps, SHAP, LOCO, prevalence PR-AUC, split
  checks, sequence models).
- `tests/` (pytest), `CHANGELOG.md`, `experimental/README.md`.
- Release files `windows_meta.csv`, `split_train_mask.npy`, `split_test_mask.npy`.

### Removed
- Synthetic variants (`synth_strong`, `synth_stealth`) and their scripts;
  TSTR, mixed-training and synthetic-fidelity experiments. The TimeGAN
  generator moved to `experimental/` and is not part of the release.
- Jupyter notebooks (`notebooks/`), `datasets/README.md`, `scripts/README.md`
  (content merged into the main README).

### Known issues
- MGEN features (f19–f24) are constant because Stage 1 cannot align the
  date-less `mgen.csv` timestamps with the KPM index, and uses the first UE's
  `mgen.csv` for every UE (see README). The source logs are not empty; the fix
  is planned for a later dataset version.
- The source KPMs are sampled every 250 ms; the release uses a 100 ms grid
  with linear interpolation (about 60 % of the grid points interpolated,
  about 20 measured samples per 5 s window, 2.5 s stride).

## v1.0.0

Initial release.
