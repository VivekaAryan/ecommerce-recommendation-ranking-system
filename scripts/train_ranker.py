#!/usr/bin/env python3
"""Train ranking models."""

import os
import sys
from pathlib import Path

# OpenMP must be configured before PyTorch/FAISS import (macOS crash otherwise).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Allow running before editable install: `python scripts/train_ranker.py`
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recsys.cli import train_ranker

if __name__ == "__main__":
    train_ranker()
