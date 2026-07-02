"""MMR diversity re-ranking."""

from __future__ import annotations

import numpy as np


def maximal_marginal_relevance(
    item_ids: list[str],
    relevance_scores: dict[str, float],
    embeddings: dict[str, np.ndarray],
    top_k: int = 10,
    lambda_param: float = 0.7,
) -> list[str]:
    if not item_ids:
        return []
    selected: list[str] = []
    candidates = item_ids.copy()
    while candidates and len(selected) < top_k:
        best_item = None
        best_score = -np.inf
        for item_id in candidates:
            relevance = relevance_scores.get(item_id, 0.0)
            if not selected:
                diversity_penalty = 0.0
            else:
                sims = [
                    float(np.dot(embeddings[item_id], embeddings[s]))
                    for s in selected
                    if item_id in embeddings and s in embeddings
                ]
                diversity_penalty = max(sims) if sims else 0.0
            mmr = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            if mmr > best_score:
                best_score = mmr
                best_item = item_id
        if best_item is None:
            break
        selected.append(best_item)
        candidates.remove(best_item)
    return selected


def intra_list_diversity(item_ids: list[str], embeddings: dict[str, np.ndarray]) -> float:
    if len(item_ids) < 2:
        return 0.0
    sims = []
    for i, a in enumerate(item_ids):
        for b in item_ids[i + 1 :]:
            if a in embeddings and b in embeddings:
                sims.append(float(np.dot(embeddings[a], embeddings[b])))
    if not sims:
        return 0.0
    return 1.0 - float(np.mean(sims))
