"""Fleetweaver: reproducible, laptop-scale experiments built on PARCO."""

import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib")
)
