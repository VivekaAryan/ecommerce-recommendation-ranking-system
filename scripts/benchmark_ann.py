#!/usr/bin/env python3
"""Benchmark ANN recall/latency tradeoff."""

from __future__ import annotations

import numpy as np

from recsys.config import get_base_config
from recsys.data.download import prepare_dataset
from recsys.data.splits import load_splits
from recsys.experiments.tracker import ExperimentTracker
from recsys.retrieval.ann_index import benchmark_ann
from recsys.utils import ensure_dir


def main() -> None:
    cfg = get_base_config()
    prepare_dataset(use_synthetic=True, synthetic_size=5_000)
    processed = cfg.resolve_path(cfg.paths.processed_dir)
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
