#!/usr/bin/env python3
"""Download Amazon Reviews 2023 multi-category subset."""

import sys
from pathlib import Path

# Allow running before editable install: `python scripts/download_data.py`
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recsys.cli import download_data

if __name__ == "__main__":
    download_data()
