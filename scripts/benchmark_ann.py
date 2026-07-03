#!/usr/bin/env python3
"""Benchmark ANN recall/latency tradeoff."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running before editable install: `python scripts/benchmark_ann.py`
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from recsys.config import get_base_config
from recsys.data.splits import load_splits
from recsys.experiments.tracker import ExperimentTracker
from recsys.retrieval.ann_index import benchmark_ann
from recsys.utils import ensure_dir


def main() -> None:
    cfg = get_base_config()
    processed = cfg.resolve_path(cfg.paths.processed_dir)
    if not (processed / "interactions.parquet").exists():
        raise FileNotFoundError("Dataset not prepared. Run: python scripts/download_data.py")
    interactions, items, train, _, _ = load_splits(processed)
    rng = np.random.default_rng(cfg.seed)
    item_ids = items["item_id"].tolist()
    embeddings = rng.normal(size=(len(item_ids), 32)).astype(np.float32)
    query = embeddings[:10]
    truth = [[item_ids[i]] for i in range(10)]
    results = benchmark_ann(embeddings, item_ids, query, truth, k=10)
    tracker = ExperimentTracker()
    with tracker.run("benchmark_ann"):
        for result in results:
            tracker.log_metrics(
                {
                    f"{result.index_type}_latency_ms": result.latency_ms,
                    f"{result.index_type}_recall@{result.k}": result.recall_at_k,
                }
            )
            print(result)
    out = ensure_dir(cfg.resolve_path(cfg.paths.models_dir) / "retrieval")
    print(f"Benchmark complete. Artifacts dir: {out}")


if __name__ == "__main__":
    main()
