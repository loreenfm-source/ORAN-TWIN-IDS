"""Attack injection (Stage 3, paper Table 4).

Each window is processed independently, in row order, with a single
``numpy.random.RandomState(seed)``. With probability ``ATTACK_FRACTION`` the
window becomes an attack: a contiguous segment of one to two target signals is
perturbed by a trapezoidal ramp envelope (linear rise over the first third,
plateau, linear fall over the last third). Telemetry outside the segment, and
all non-target signals, are left untouched; no blending with other windows is
performed.

The order of the RNG calls is part of the dataset definition. Per window:
``rand`` (attack or not); for attacks additionally ``choice`` (type),
``uniform`` (duration), ``randint`` (start) and one ``uniform`` per perturbed
signal (DoS: bitrate then PRB; Scan: packets; PHY: SINR then MCS). Changing
this order changes which windows are attacked and therefore the released
class counts (paper Table 5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import ATTACK_TYPES, INJECTION_TARGETS

ATTACK_FRACTION = 0.25
DEFAULT_SEED = 42


@dataclass(frozen=True)
class InjectionParams:
    """Ranges of the uniform distributions of paper Table 4.

    ``duration`` is a fraction of the window length T; all magnitudes are peak
    amplitudes in the StandardScaler-normalised feature space. PHY magnitudes
    are subtracted.
    """

    duration: tuple[float, float]
    dos_brate: tuple[float, float]
    dos_prb: tuple[float, float]
    scan_pkts: tuple[float, float]
    phy_sinr: tuple[float, float]
    phy_mcs: tuple[float, float]


VARIANTS: dict[str, InjectionParams] = {
    "strong": InjectionParams(
        duration=(0.2, 0.6), dos_brate=(2.0, 4.0), dos_prb=(1.0, 3.0),
        scan_pkts=(2.0, 5.0), phy_sinr=(2.0, 4.0), phy_mcs=(1.0, 2.0),
    ),
    "stealth": InjectionParams(
        duration=(0.1, 0.35), dos_brate=(0.4, 0.9), dos_prb=(0.3, 0.7),
        scan_pkts=(0.4, 0.9), phy_sinr=(0.4, 0.9), phy_mcs=(0.2, 0.5),
    ),
}


def ramp_envelope(length: int) -> np.ndarray:
    """Trapezoidal envelope of ``length`` samples with rise/fall of ``length // 3``."""
    env = np.ones(length, np.float32)
    k = max(1, length // 3)
    env[:k] = np.linspace(0, 1, k)
    env[-k:] = np.linspace(1, 0, k)
    return env


def inject_window(window: np.ndarray, rng: np.random.RandomState,
                  params: InjectionParams) -> str:
    """Possibly inject an attack into ``window`` (shape ``(T, F)``) in place.

    Returns the attack type (``'benign'``, ``'dos'``, ``'scan'`` or ``'phy'``).
    """
    if rng.rand() >= ATTACK_FRACTION:
        return "benign"

    n_steps, n_feat = window.shape
    attack = str(rng.choice(list(ATTACK_TYPES)))
    dur = int(n_steps * rng.uniform(*params.duration))
    start = rng.randint(0, n_steps - dur)
    end = start + dur

    env = ramp_envelope(end - start)[:, None]
    delta = np.zeros((end - start, n_feat), np.float32)
    if attack == "dos":
        delta[:, INJECTION_TARGETS["tx_brate"]] = rng.uniform(*params.dos_brate)
        delta[:, INJECTION_TARGETS["sum_granted_prbs"]] = rng.uniform(*params.dos_prb)
    elif attack == "scan":
        delta[:, INJECTION_TARGETS["tx_pkts"]] = rng.uniform(*params.scan_pkts)
    else:  # phy
        delta[:, INJECTION_TARGETS["ul_sinr"]] = -rng.uniform(*params.phy_sinr)
        delta[:, INJECTION_TARGETS["ul_mcs"]] = -rng.uniform(*params.phy_mcs)

    window[start:end] += env * delta
    return attack


def inject_windows(windows: np.ndarray, params: InjectionParams,
                   seed: int = DEFAULT_SEED, out: np.ndarray | None = None,
                   progress: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Inject attacks into every window of ``windows`` (shape ``(N, T, F)``).

    The input is never modified: each window is copied before injection.
    Results are written to ``out`` (e.g. a ``np.memmap``) or to a new array.

    Returns ``(injected_windows, attack_types)`` where ``attack_types`` is an
    object array of strings of length N.
    """
    n = windows.shape[0]
    if out is None:
        out = np.empty(windows.shape, np.float32)
    rng = np.random.RandomState(seed)
    attack_types = np.empty(n, dtype=object)

    iterator = range(n)
    if progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="Injecting attacks", mininterval=5)
    for i in iterator:
        w = np.array(windows[i], dtype=np.float32, copy=True)
        attack_types[i] = inject_window(w, rng, params)
        out[i] = w
    return out, attack_types
