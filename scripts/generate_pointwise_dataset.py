#!/usr/bin/env python3
"""Stage 1 - build the point-wise telemetry table from the source dataset.

Traverses ``<root>/cluster_*/slicing_*/scheduling_*/RESERVATION-*`` of the Open
RAN Commercial Traffic Twinning Dataset (Bonati et al., 2024), reads the per-UE
KPM files ``bs/<IMSI>_metrics.csv``, resamples each UE's series onto a regular
grid, left-joins MGEN application statistics and appends one row per UE per
time step to ``pointwise_dataset.csv``.

Output columns (29): the 19 KPM and 6 MGEN feature columns of
``oran_twin_ids.features.RAW_FEATURES``, then ``reservation_id``, ``imsi``,
``timestamp`` (ISO 8601) and ``label`` (0 unless a label manifest is given).

Notes for users
---------------
* The source KPMs are sampled every 250 ms. ``--resample_freq`` sets the grid:
  ``100ms`` (default) reproduces the released data, in which about 60 % of the
  grid points are linearly interpolated between measured samples; ``250ms``
  keeps the native cadence; ``auto`` uses the median timestamp spacing of each
  file. A file with a single sample is binned to the second.
* The source dataset reuses the same eight IMSIs in every reservation, so rows
  must be grouped by (reservation_id, imsi), never by IMSI alone.
* Known issue (kept so that the released data can be reproduced): ``mgen.csv``
  timestamps are times of day without a date. ``parse_ts`` resolves them to the
  current date, so they never match the KPM index and every MGEN column is
  zero-filled (``mgen_missing`` = 1). In addition, the first ``ue_*/mgen.csv``
  of a reservation is used for all its UEs. Consequently f19-f24 are constant.

The run is resumable: processed reservations are recorded in
``manifest_progress.json`` and unreadable inputs are logged to
``bad_files.log``.

Example::

    python scripts/generate_pointwise_dataset.py --root_dir /data/dataset-kpm \
        --output_dir outputs/pointwise
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oran_twin_ids.features import KPM_FEATURES, MGEN_FEATURES  # noqa: E402

log = logging.getLogger("pointwise")

DEFAULT_RESAMPLE_FREQ = "100ms"  # grid of the released data
SINGLE_SAMPLE_FREQ = "1s"        # bin of a file with a single sample (as released)
MGEN_RESAMPLE_FREQ = "1s"        # MGEN aggregation interval
INTERPOLATE_LIMIT = 5            # max consecutive gaps filled by linear interpolation
SMALL_PACKET_BYTES = 100
META_COLUMNS = ["reservation_id", "imsi", "timestamp", "label"]
OUTPUT_COLUMNS = KPM_FEATURES + MGEN_FEATURES + META_COLUMNS

_CANDIDATE_DT_FORMATS = [
    "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S",
    "%d/%m/%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%d-%b-%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
]


def log_bad_file(path, reason: str, log_path: Path) -> None:
    """Append ``path<TAB>reason`` to the bad-files log."""
    try:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"{path}\t{reason}\n")
    except OSError:
        pass


def _try_vectorized_format(series: pd.Series, fmt: str):
    try:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
    except (ValueError, TypeError):
        return None
    return parsed if parsed.notna().sum() / max(1, len(series)) > 0.80 else None


def _is_numeric_series(series: pd.Series) -> bool:
    if np.issubdtype(series.dtype, np.number):
        return True
    s = series.dropna().astype(str)
    return len(s) > 0 and s.str.match(r"^\d+(\.\d+)?$").mean() > 0.9


def parse_ts(series: pd.Series) -> pd.Series:
    """Parse timestamps: epoch seconds/milliseconds or a set of string formats."""
    if np.issubdtype(series.dtype, np.datetime64):
        return pd.to_datetime(series)
    if _is_numeric_series(series):
        vals = pd.to_numeric(series, errors="coerce")
        median_val = np.nanmedian(vals)
        if not np.isnan(median_val):
            unit = "ms" if median_val > 1e11 else "s"
            return pd.to_datetime(vals.astype("Int64"), unit=unit, errors="coerce")
    s = series.astype(str).str.strip()
    sample = s.dropna().head(500)
    for fmt in _CANDIDATE_DT_FORMATS:
        if _try_vectorized_format(sample, fmt) is not None:
            return pd.to_datetime(s, format=fmt, errors="coerce")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Could not infer format")
        return pd.to_datetime(s, errors="coerce")


def median_spacing_freq(index: pd.DatetimeIndex) -> str:
    """Pandas frequency string from the median spacing of ``index`` (any datetime unit)."""
    ns = np.asarray(index.values, dtype="datetime64[ns]").astype(np.int64)
    median_ms = max(1, int(round(np.median(np.diff(ns)) / 1e6)))
    if median_ms < 1000:
        return f"{median_ms}ms"
    return f"{int(round(median_ms / 1000.0))}s"


def resample_freq_for(index: pd.DatetimeIndex, resample_freq: str) -> str:
    """Grid of one UE series: ``resample_freq``, or the median spacing for ``auto``."""
    if len(index) < 2:
        return SINGLE_SAMPLE_FREQ
    return median_spacing_freq(index) if resample_freq == "auto" else resample_freq


def aggregate_mgen(mgen_path, resample_index: pd.DatetimeIndex, resample_freq: str,
                   bad_files_log: Path) -> pd.DataFrame:
    """Per-interval MGEN statistics reindexed onto ``resample_index``.

    Missing, empty or unreadable files give zeros with ``mgen_missing = 1``.
    """
    def missing():
        df = pd.DataFrame(0, index=resample_index, columns=MGEN_FEATURES)
        df["mgen_missing"] = 1
        return df

    if mgen_path is None or not os.path.exists(mgen_path) or os.path.getsize(mgen_path) == 0:
        return missing()
    try:
        m = pd.read_csv(mgen_path)
        if m.shape[0] == 0:
            raise ValueError("empty mgen")
        tscol = next((c for c in ["Timestamp_recv", "Received Time", "Timestamp_send", "Sent Time"]
                      if c in m.columns), m.columns[0])
        m["_ts"] = parse_ts(m[tscol])
        m = m.dropna(subset=["_ts"]).set_index("_ts").sort_index()
        payload_cols = [c for c in m.columns if "payload" in c.lower()]
        payload = payload_cols[0] if payload_cols else None

        def agg_fn(dfw):
            total_pkts = len(dfw)
            total_payload = float(dfw[payload].sum()) if payload in dfw.columns else 0.0
            return pd.Series({
                "mgen_flow_count": int(dfw["Flow ID"].nunique()) if "Flow ID" in dfw.columns else 0,
                "mgen_total_pkts": total_pkts,
                "mgen_total_payload": total_payload,
                "mgen_avg_pkt_size": total_payload / total_pkts if total_pkts > 0 else 0.0,
                "mgen_small_pkt_ratio": (float((dfw[payload] <= SMALL_PACKET_BYTES).sum() / total_pkts)
                                         if payload and total_pkts > 0 else 0.0),
            })

        agg = m.resample(resample_freq).apply(agg_fn).fillna(0)
        agg = agg.reindex(resample_index, fill_value=0)
        traffic = agg[["mgen_flow_count", "mgen_total_pkts", "mgen_total_payload"]].sum(axis=1)
        agg["mgen_missing"] = (traffic == 0).astype(int)
        return agg
    except Exception as exc:  # malformed third-party CSV: record and continue
        log_bad_file(mgen_path, f"mgen_read_error:{str(exc)[:200]}", bad_files_log)
        return missing()


def load_ue_series(metrics_path: str, bad_files_log: Path, mgen_path=None,
                   resample_freq: str = DEFAULT_RESAMPLE_FREQ) -> pd.DataFrame | None:
    """Resampled KPM series of one UE joined with its MGEN statistics."""
    if not os.path.exists(metrics_path) or os.path.getsize(metrics_path) == 0:
        log_bad_file(metrics_path, "missing_or_empty", bad_files_log)
        return None
    try:
        df = pd.read_csv(metrics_path, low_memory=False)
    except Exception as exc:  # malformed third-party CSV
        log_bad_file(metrics_path, f"read_error:{type(exc).__name__}:{str(exc)[:200]}", bad_files_log)
        return None
    if df.shape[0] == 0:
        log_bad_file(metrics_path, "no_rows_after_read", bad_files_log)
        return None
    tscol = "Timestamp" if "Timestamp" in df.columns else df.columns[0]
    try:
        df["_ts"] = parse_ts(df[tscol])
    except Exception as exc:
        log_bad_file(metrics_path, f"timestamp_parse_error:{str(exc)[:200]}", bad_files_log)
        return None
    df = df.dropna(subset=["_ts"]).set_index("_ts").sort_index()
    if df.shape[0] == 0:
        log_bad_file(metrics_path, "all_timestamps_nan", bad_files_log)
        return None
    features = [f for f in KPM_FEATURES if f in df.columns]
    if not features:
        return None  # not a per-UE file (e.g. enb_metrics.csv)
    selected = df[features].apply(pd.to_numeric, errors="coerce")
    resampled = selected.resample(resample_freq_for(df.index, resample_freq)).mean()
    resampled = resampled.interpolate(limit=INTERPOLATE_LIMIT).ffill().bfill()
    mgen = aggregate_mgen(mgen_path, resampled.index, MGEN_RESAMPLE_FREQ, bad_files_log)
    return resampled.join(mgen, how="left").fillna(0)


def find_ue_metric_files(bs_dir: str, reservation_dir: str) -> list[str]:
    """Per-UE KPM files of a reservation (``<IMSI>_metrics.csv``; never ``enb_metrics.csv``)."""
    cand = sorted(glob.glob(os.path.join(bs_dir, "*_metrics.csv")))
    ue_files = [f for f in cand if re.match(r"^\d+_metrics\.csv$", os.path.basename(f))]
    if ue_files:
        return ue_files
    ue_files = [f for f in cand
                if not os.path.basename(f).lower().startswith(("enb", "enodeb"))]
    if ue_files:
        return ue_files
    return sorted(glob.glob(os.path.join(reservation_dir, "ue_*", "ue_metrics.csv")))


def load_label_manifest(path):
    """Optional CSV/JSON with ``reservation_id, start_ts, end_ts, label`` rows."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"label manifest not found: {p}")
    if p.suffix.lower() == ".json":
        return pd.DataFrame(json.loads(p.read_text(encoding="utf-8")))
    df = pd.read_csv(p)
    for c in ("start_ts", "end_ts"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def label_series(index: pd.DatetimeIndex, reservation_id: str, manifest) -> np.ndarray:
    """Point labels of one series from the optional manifest (default 0)."""
    labels = np.zeros(len(index), dtype=int)
    if manifest is None:
        return labels
    for _, r in manifest[manifest["reservation_id"] == reservation_id].iterrows():
        start, end = pd.to_datetime(r.get("start_ts")), pd.to_datetime(r.get("end_ts"))
        if pd.isna(start) or pd.isna(end):
            continue
        labels[(index >= start) & (index <= end)] = int(r.get("label", 1))
    return labels


def ue_rows(series: pd.DataFrame, reservation_id: str, imsi: str, labels: np.ndarray) -> pd.DataFrame:
    """Rows of one UE in the fixed output schema (all features as float64)."""
    out = series.reindex(columns=KPM_FEATURES + MGEN_FEATURES).astype(np.float64)
    out["reservation_id"] = reservation_id
    out["imsi"] = imsi
    out["timestamp"] = [t.isoformat() for t in series.index]
    out["label"] = labels
    return out.reset_index(drop=True)[OUTPUT_COLUMNS]


def find_reservations(root_dir: Path) -> list[str]:
    """Reservation directories in sorted path order (cluster, slicing, scheduling, id)."""
    pattern = os.path.join(root_dir, "cluster_*", "slicing_*", "scheduling_*", "RESERVATION-*")
    matches = sorted(glob.glob(pattern))
    if matches:
        return matches
    log.warning("no cluster_*/slicing_*/scheduling_*/RESERVATION-* directories; "
                "falling back to any directory named *reservation*")
    return sorted(os.path.join(r, d) for r, ds, _ in os.walk(root_dir)
                  for d in ds if "reservation" in d.lower())


def build_pointwise(root_dir: Path, out_dir: Path, label_manifest=None,
                    max_reservations: int | None = None,
                    resample_freq: str = DEFAULT_RESAMPLE_FREQ) -> Path:
    """Run Stage 1 and return the path of ``pointwise_dataset.csv``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "pointwise_dataset.csv"
    progress_json = out_dir / "manifest_progress.json"
    bad_files_log = out_dir / "bad_files.log"
    manifest = load_label_manifest(label_manifest)

    progress = {"processed": []}
    if progress_json.exists():
        progress = json.loads(progress_json.read_text(encoding="utf-8"))
        log.info("Resuming: %d reservations already processed", len(progress["processed"]))
    done = {p["reservation"] for p in progress["processed"]}
    header_written = out_csv.exists()
    if header_written and not done:
        raise FileExistsError(f"{out_csv} exists without a progress manifest; "
                              "move it away before starting a new run")

    reservations = find_reservations(root_dir)
    log.info("%d candidate reservations found", len(reservations))
    if max_reservations:
        reservations = reservations[:max_reservations]

    for res in tqdm(reservations, desc="reservations"):
        reservation_id = os.path.basename(res)
        if reservation_id in done:
            continue
        bs_dir = next((os.path.join(res, d) for d in ("bs", "BS")
                       if os.path.isdir(os.path.join(res, d))), res)
        ue_files = find_ue_metric_files(bs_dir, res)
        # Known issue, kept for reproducibility: one mgen.csv is used for all UEs.
        mgen_candidates = glob.glob(os.path.join(res, "ue_*", "mgen.csv"))
        mgen_path = mgen_candidates[0] if mgen_candidates else None

        frames = []
        for ue_file in ue_files:
            imsi = os.path.basename(ue_file).split("_")[0]
            series = load_ue_series(ue_file, bad_files_log, mgen_path, resample_freq)
            if series is None or series.shape[0] == 0:
                continue
            missing = [c for c in KPM_FEATURES if c not in series.columns]
            if missing:
                log_bad_file(ue_file, f"missing_features:{missing}", bad_files_log)
            frames.append(ue_rows(series, reservation_id, imsi,
                                  label_series(series.index, reservation_id, manifest)))
        n_rows = sum(len(f) for f in frames)
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(
                out_csv, mode="a" if header_written else "w", header=not header_written, index=False)
            header_written = True
        if not ue_files:
            tqdm.write(f"[warn] no UE metric files in {reservation_id}")
        progress["processed"].append({"reservation": reservation_id, "processed_ue_files": len(frames),
                                      "timestamp": time.time(), "rows_written": n_rows})
        progress_json.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        done.add(reservation_id)
    log.info("Done: %s", out_csv)
    return out_csv


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Stage 1: build the point-wise telemetry table.")
    p.add_argument("--root_dir", type=Path, required=True,
                   help="root of the source dataset (contains cluster_1, cluster_2, cluster_3)")
    p.add_argument("--output_dir", type=Path, default=Path("outputs/pointwise"))
    p.add_argument("--label_manifest", type=Path, default=None,
                   help="optional CSV/JSON of labelled time ranges (default: all labels 0)")
    p.add_argument("--max_reservations", type=int, default=None,
                   help="process only the first N reservations (for testing)")
    p.add_argument("--resample_freq", default=DEFAULT_RESAMPLE_FREQ,
                   help="grid step: 100ms (default, as released), 250ms (native cadence of "
                        "the source KPMs), any pandas frequency, or 'auto' (median spacing "
                        "of each file)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not args.root_dir.is_dir():
        p.error(f"--root_dir is not a directory: {args.root_dir}")
    if args.resample_freq != "auto":
        try:
            pd.tseries.frequencies.to_offset(args.resample_freq)
        except ValueError:
            p.error(f"--resample_freq is not a pandas frequency or 'auto': {args.resample_freq}")
    build_pointwise(args.root_dir, args.output_dir, args.label_manifest, args.max_reservations,
                    args.resample_freq)


if __name__ == "__main__":
    main()
