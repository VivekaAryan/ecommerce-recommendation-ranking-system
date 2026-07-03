#!/usr/bin/env python3
"""Build batch feature tables."""

import sys
from pathlib import Path

# Allow running before editable install: `python scripts/build_features.py`
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recsys.cli import build_features

if __name__ == "__main__":
    build_features()
