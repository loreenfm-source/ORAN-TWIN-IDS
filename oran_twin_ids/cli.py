"""Small helpers shared by the command-line scripts."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from pathlib import Path


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")


def require_file(parser: argparse.ArgumentParser, path: Path | None, flag: str) -> None:
    """Exit with a clear message if ``path`` (given via ``flag``) is not a file."""
    if path is not None and not Path(path).is_file():
        parser.error(f"{flag}: file not found: {path}")


def write_run_info(output_dir: Path, args: argparse.Namespace) -> None:
    """Record the arguments and library versions of a run in ``run_info.json``."""
    versions = {}
    for mod in ("numpy", "pandas", "sklearn", "xgboost", "lightgbm", "shap", "umap", "torch"):
        m = sys.modules.get(mod)
        if m is not None:
            versions[mod] = getattr(m, "__version__", "?")
    info = {
        "script": Path(sys.argv[0]).name,
        "arguments": {k: str(v) for k, v in vars(args).items()},
        "python": platform.python_version(),
        "libraries": versions,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
