"""Stages 3 and 4 for one variant: inject attacks, then extract features."""

from __future__ import annotations

import argparse
import logging
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from .extraction import write_feature_csv
from .features import N_FEATURES
from .injection import DEFAULT_SEED, VARIANTS, inject_windows

log = logging.getLogger(__name__)


def open_windows(path: Path, workdir: Path) -> tuple[np.ndarray, Path | None]:
    """Memory-map ``windows_scaled.npz`` (key ``windows``) or a ``.npy`` file.

    ``np.load(..., mmap_mode='r')`` cannot memory-map a member of a zip
    archive, so an NPZ is first extracted to ``workdir/_windows_scaled.npy``
    (about 18 GB for the full dataset). Returns the array and the path of the
    extracted temporary file (``None`` for ``.npy`` input).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"windows file not found: {path}")
    tmp = None
    if path.suffix == ".npy":
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
    else:
        with zipfile.ZipFile(path) as zf:
            if "windows.npy" not in zf.namelist():
                raise KeyError(f"{path} has no 'windows' array")
            tmp = Path(workdir) / "_windows_scaled.npy"
            log.info("Extracting %s to %s", path.name, tmp)
            with zf.open("windows.npy") as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 24)
        arr = np.load(tmp, mmap_mode="r", allow_pickle=False)
    if arr.ndim != 3 or arr.shape[2] != N_FEATURES or arr.dtype != np.float32:
        raise ValueError(f"expected float32 windows (N, T, {N_FEATURES}), got {arr.dtype} {arr.shape}")
    return arr, tmp


def generate_variant(variant: str, argv=None) -> None:
    """Command-line entry point used by ``generate_real_<variant>_dataset.py``."""
    p = argparse.ArgumentParser(
        description=f"Stages 3-4: inject {variant} attacks and write real_{variant}.csv.")
    p.add_argument("--input", type=Path, required=True,
                   help="windows_scaled.npz from Stage 2 (or an extracted .npy)")
    p.add_argument("--output_dir", type=Path, default=Path(f"outputs/real_{variant}"))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--keep_injected", action="store_true",
                   help="keep the injected windows as windows_injected.npy")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    params = VARIANTS[variant]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    windows, extracted = open_windows(args.input, args.output_dir)
    injected_path = args.output_dir / "windows_injected.npy"
    injected = np.lib.format.open_memmap(injected_path, mode="w+", dtype=np.float32,
                                         shape=windows.shape)
    log.info("Injecting %s attacks into %s windows (seed %d)", variant,
             f"{len(windows):,}", args.seed)
    injected, attack_types = inject_windows(windows, params, args.seed, out=injected, progress=True)
    injected.flush()

    out_csv = args.output_dir / f"real_{variant}.csv"
    write_feature_csv(injected, attack_types, out_csv, progress=True)
    counts = Counter(attack_types)
    log.info("Wrote %s | classes: %s", out_csv, {k: counts.get(k, 0) for k in
                                                 ("benign", "dos", "scan", "phy")})
    del injected, windows
    if not args.keep_injected:
        injected_path.unlink()
    if extracted is not None:
        extracted.unlink()
