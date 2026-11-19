#!/usr/bin/env python3
"""Stages 3-4 for the real_stealth variant (paper Table 4, stealth column).

Reads the benign scaled windows, injects stealth attacks with the ramp envelope
(seed 42 by default), computes mean/std/p95 per signal and writes the
77-column real_stealth.csv. See oran_twin_ids/injection.py for the exact
procedure.

Example::

    python scripts/generate_real_stealth_dataset.py         --input outputs/windows/windows_scaled.npz --output_dir outputs/real_stealth
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oran_twin_ids.variant import generate_variant  # noqa: E402

if __name__ == "__main__":
    generate_variant("stealth")
