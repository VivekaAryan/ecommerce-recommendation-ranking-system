"""Serving pipeline with latency profiling."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from recsys.config import load_yaml


@dataclass
class StageLatency:
    retrieval_ms: float = 0.0
    prerank_ms: float = 0.0
    ranking_ms: float = 0.0
    reranking_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.retrieval_ms + self.prerank_ms + self.ranking_ms + self.reranking_ms


@dataclass
class ServingPipeline:
    retriever: object
    preranker: object
    ranker: object
    reranker: object
    budget_ms: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.budget_ms:
            sim_cfg = load_yaml("simulator.yaml")
            self.budget_ms = sim_cfg["latency_budget_ms"]

    def recommend(self, user_id: str, context: dict) -> tuple[list[str], StageLatency]:
        latency = StageLatency()

        start = time.perf_counter()
        candidates = self.retriever.retrieve(user_id, context)
        latency.retrieval_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        preranked = self.preranker.prerank(candidates, context)
        latency.prerank_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        ranked = self.ranker.rank(preranked, context)
        latency.ranking_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        slate = self.reranker.rerank(ranked, context)
        latency.reranking_ms = (time.perf_counter() - start) * 1000

        return slate, latency

    def within_budget(self, latency: StageLatency) -> dict[str, bool]:
        return {
            "retrieval": latency.retrieval_ms <= self.budget_ms["retrieval"],
            "prerank": latency.prerank_ms <= self.budget_ms.get("prerank", 10),
            "ranking": latency.ranking_ms <= self.budget_ms["ranking"],
            "reranking": latency.reranking_ms <= self.budget_ms["reranking"],
            "total": latency.total_ms <= self.budget_ms["total"],
        }
