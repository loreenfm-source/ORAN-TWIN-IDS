# Experimental: TimeGAN synthetic windows (not released)

`generate_synthetic_windows_timegan.py` trains the simplified TimeGAN generator
that was evaluated as a source of synthetic telemetry (supplementary Section
S1.3). **It was not used to build the released dataset, and no synthetic
variant is released.**

Summary of the evaluation reported in the paper:

* Fidelity was measured as the ratio of the mean within-window standard
  deviation of synthetic and real windows on the five injection targets
  (f0, f3, f8, f13, f15), against a random sample of real windows from all
  clusters.
* The first generator (autoencoder pre-training + adversarial training only)
  gave ratios of 6–19×. Adding TimeGAN's supervised next-step loss and a
  within-window variance moment-matching term reduced them to 0.7–1.8× in the
  run reported in the paper.
* Attacks injected into synthetic windows with the strong parameters were not
  detectable above the trivial level (binary F1 ≤ 0.26 for all baselines, vs
  0.40 for "always attack"). The cause was not identified, so the synthetic
  windows were not released.

Notes on this script:

* The default `--eta 10` is the value in the last generator notebook available
  to the maintainers; that notebook's own fidelity printout (computed on the
  first 20,000 windows rather than a random sample) gave ratios between 0.8
  and 5.5. The configuration behind the 0.7–1.8 figure has not been confirmed.
* The fidelity reference is now a random 20,000-window sample of the whole
  archive (earlier code used the first 20,000 rows).
* A suggested diagnostic that was never completed: after injecting into the
  synthetic windows, check that the maximum absolute difference between the
  injected and the original window is > 0 for every attack row and exactly 0
  for every benign row (label/window alignment).

Usage (PyTorch and a GPU required; several hours on the full archive):

```bash
python experimental/generate_synthetic_windows_timegan.py \
    --windows data/windows_scaled.npz --output_dir outputs/timegan
```
