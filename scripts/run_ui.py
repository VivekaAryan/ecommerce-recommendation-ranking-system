#!/usr/bin/env python3
"""Start the recsys testing dashboard API server."""

import sys
from pathlib import Path

# Allow running before editable install: `python scripts/run_ui.py`
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recsys.api.server import main

if __name__ == "__main__":
    main()
