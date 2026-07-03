#!/usr/bin/env python3
"""Run multi-cycle feedback loop simulation."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running before editable install: `python scripts/run_feedback_loop.py`
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from recsys.config import get_base_config, load_yaml
from recsys.data.splits import load_splits
from recsys.evaluation.feedback_loop import measure_feedback_loop
from recsys.simulator.runner import build_default_runner
from recsys.utils import ensure_dir


def main() -> None:
    cfg = get_base_config()
    processed = cfg.resolve_path(cfg.paths.processed_dir)
    if not (processed / "interactions.parquet").exists():
        raise FileNotFoundError("Dataset not prepared. Run: python scripts/download_data.py")
    interactions, items, _, _, _ = load_splits(processed)
    sim_cfg = load_yaml("simulator.yaml")
    logs_per_cycle = []
    for cycle in range(3):
        log_dir = ensure_dir(cfg.resolve_path(cfg.paths.simulator_logs_dir) / f"cycle_{cycle}")
        runner = build_default_runner(items, interactions, log_dir)
        runner.run(sim_cfg["simulation"]["num_days"], sim_cfg["simulation"]["sessions_per_day"])
        logs_per_cycle.append(runner.logger.to_dataframe())
    metrics = measure_feedback_loop(logs_per_cycle, catalog_size=len(items))
    report = pd.DataFrame([m.__dict__ for m in metrics])
    out = cfg.resolve_path(cfg.paths.simulator_logs_dir) / "feedback_loop_report.csv"
    report.to_csv(out, index=False)
    print(report)
    print(f"Saved feedback loop report to {out}")


if __name__ == "__main__":
    main()
