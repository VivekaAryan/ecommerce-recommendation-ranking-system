"""Build item feature vectors for retrieval models."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from recsys.features.embeddings import EMBEDDING_DIM

EMBEDDING_COLS = [f"content_emb_{i}" for i in range(EMBEDDING_DIM)]


def build_item_feature_matrix(
    items: pd.DataFrame,
    item_features: pd.DataFrame | None = None,
) -> np.ndarray:
    """Build per-item side feature matrix aligned with items row order."""
    features_df = items.copy()
    if item_features is not None:
        features_df = features_df.merge(item_features, on="item_id", how="left", suffixes=("", "_feat"))

    matrix: list[list[float]] = []
    for row in features_df.itertuples(index=False):
        price = 0.0 if pd.isna(getattr(row, "price", np.nan)) else float(row.price)
        category = getattr(row, "main_category", None) or getattr(row, "category", "Unknown")
        cat_code = hash(str(category)) % 1000 / 1000.0
        avg_rating = getattr(row, "avg_rating", getattr(row, "average_rating", 0.0))
        avg_rating = 0.0 if pd.isna(avg_rating) else float(avg_rating) / 5.0
        link_degree = getattr(row, "link_degree", 0.0)
        link_degree = 0.0 if pd.isna(link_degree) else float(link_degree)
        log_rating_number = getattr(row, "log_rating_number", 0.0)
        log_rating_number = 0.0 if pd.isna(log_rating_number) else float(log_rating_number)

        emb = []
        for col in EMBEDDING_COLS:
            val = getattr(row, col, 0.0)
            emb.append(0.0 if pd.isna(val) else float(val))

        if not emb:
            emb = [0.0] * EMBEDDING_DIM

        avg_helpful_vote = getattr(row, "avg_helpful_vote", 0.0)
        avg_helpful_vote = 0.0 if pd.isna(avg_helpful_vote) else math.log1p(float(avg_helpful_vote)) / 5.0

        matrix.append(
            [
                price / 500.0,
                cat_code,
                avg_rating,
                math.log1p(link_degree) / 5.0,
                log_rating_number,
                avg_helpful_vote,
                *emb,
            ]
        )

    return np.array(matrix, dtype=np.float32)


def item_feature_dim() -> int:
    return 6 + EMBEDDING_DIM
