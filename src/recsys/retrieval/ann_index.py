"""FAISS ANN index build and search."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np


@dataclass
class ANNBenchmarkResult:
    index_type: str
    latency_ms: float
    recall_at_k: float
    k: int


class ANNIndex:
    def __init__(self, embedding_dim: int, index_type: str = "hnsw") -> None:
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index: faiss.Index | None = None
        self.item_ids: list[str] = []

    def build(self, embeddings: np.ndarray, item_ids: list[str], **kwargs) -> None:
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            nlist = kwargs.get("nlist", 100)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings)
            self.index.nprobe = kwargs.get("nprobe", 10)
        else:
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, kwargs.get("m", 32))
            self.index.hnsw.efConstruction = kwargs.get("ef_construction", 200)
            self.index.hnsw.efSearch = kwargs.get("ef_search", 64)
        self.index.add(embeddings)
        self.item_ids = item_ids

    def search(self, query_embeddings: np.ndarray, top_k: int = 500) -> tuple[list[list[str]], np.ndarray]:
        if self.index is None:
            raise RuntimeError("Index not built")
        query_embeddings = query_embeddings.astype(np.float32)
        faiss.normalize_L2(query_embeddings)
        scores, indices = self.index.search(query_embeddings, top_k)
        results = []
        for row in indices:
            results.append([self.item_ids[i] for i in row if i >= 0])
        return results, scores

    def save(self, path: Path) -> None:
        if self.index is None:
            raise RuntimeError("Index not built")
        faiss.write_index(self.index, str(path))
        np.save(path.with_suffix(".ids.npy"), np.array(self.item_ids, dtype=object))

    def load(self, path: Path) -> None:
        self.index = faiss.read_index(str(path))
        self.item_ids = np.load(path.with_suffix(".ids.npy"), allow_pickle=True).tolist()


def benchmark_ann(
    embeddings: np.ndarray,
    item_ids: list[str],
    query_embeddings: np.ndarray,
    ground_truth: list[list[str]],
    k: int = 50,
) -> list[ANNBenchmarkResult]:
    results: list[ANNBenchmarkResult] = []
    for index_type in ("flat", "ivf", "hnsw"):
        index = ANNIndex(embeddings.shape[1], index_type=index_type)
        index.build(embeddings, item_ids)
        start = time.perf_counter()
        retrieved, _ = index.search(query_embeddings, top_k=k)
        latency_ms = (time.perf_counter() - start) / len(query_embeddings) * 1000
        recalls = []
        for pred, truth in zip(retrieved, ground_truth, strict=False):
            if not truth:
                continue
            recalls.append(len(set(pred) & set(truth)) / len(truth))
        recall = float(np.mean(recalls)) if recalls else 0.0
        results.append(ANNBenchmarkResult(index_type=index_type, latency_ms=latency_ms, recall_at_k=recall, k=k))
    return results
