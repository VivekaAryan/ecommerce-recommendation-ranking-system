#!/usr/bin/env python3
"""Start the recsys testing dashboard API server."""

import os
import sys
from pathlib import Path

# OpenMP must be configured before PyTorch/FAISS import (macOS crash otherwise).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Allow running before editable install: `python scripts/run_ui.py`
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recsys.env import configure_runtime_env

configure_runtime_env()

from recsys.api.server import main

if __name__ == "__main__":
    main()
