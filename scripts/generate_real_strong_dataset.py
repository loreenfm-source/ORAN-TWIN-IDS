#!/usr/bin/env python3
"""Stages 3-4 for the real_strong variant (paper Table 4, strong column).

Reads the benign scaled windows, injects strong attacks with the ramp envelope
(seed 42 by default), computes mean/std/p95 per signal and writes the
77-column real_strong.csv. See oran_twin_ids/injection.py for the exact
procedure.

Example::

    python scripts/generate_real_strong_dataset.py         --input outputs/windows/windows_scaled.npz --output_dir outputs/real_strong
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oran_twin_ids.variant import generate_variant  # noqa: E402

if __name__ == "__main__":
    generate_variant("strong")
