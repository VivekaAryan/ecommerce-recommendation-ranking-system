#!/usr/bin/env python3
"""Measure batch vs online feature skew."""

import sys
from pathlib import Path

# Allow running before editable install: `python scripts/measure_skew.py`
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recsys.cli import measure_skew

if __name__ == "__main__":
    measure_skew()
