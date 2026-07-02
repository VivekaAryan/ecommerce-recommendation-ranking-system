"""Fast pre-ranking stage."""

from __future__ import annotations

import pandas as pd


def prerank_candidates(
    retrieval_scores: dict[str, float],
    item_features: pd.DataFrame,
    top_k: int = 100,
) -> list[str]:
    rows = []
    for item_id, score in retrieval_scores.items():
        feat = item_features[item_features["item_id"] == item_id]
        if feat.empty:
            continue
        row = feat.iloc[0]
        combined = score + 0.01 * row.get("popularity_decay_7d", 0) - 0.005 * row.get("price_bucket", 0)
        rows.append((item_id, combined))
    rows.sort(key=lambda x: x[1], reverse=True)
    return [item_id for item_id, _ in rows[:top_k]]
