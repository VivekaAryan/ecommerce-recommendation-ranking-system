#!/usr/bin/env python3
"""Run evaluation suite."""

import sys
from pathlib import Path

# Allow running before editable install: `python scripts/evaluate.py`
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recsys.cli import evaluate

if __name__ == "__main__":
    evaluate()
