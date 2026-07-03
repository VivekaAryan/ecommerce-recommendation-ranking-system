"""LightGBM LambdaRank hybrid head."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from recsys.features.embeddings import EMBEDDING_COLS

EMBEDDING_COL_PREFIX = "content_emb_"


class LightGBMRanker:
    def __init__(self, params: dict | None = None) -> None:
        default = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "verbose": -1,
        }
        self.params = {**default, **(params or {})}
        self.model = None

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        group_sizes: list[int],
    ) -> None:
        try:
            import lightgbm as lgb
        except OSError as exc:
            raise RuntimeError(
                "LightGBM failed to load. On macOS, install OpenMP with: brew install libomp"
            ) from exc

        train_set = lgb.Dataset(features, label=labels, group=group_sizes)
        self.model = lgb.train(self.params, train_set, num_boost_round=self.params.get("n_estimators", 200))

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained")
        return self.model.predict(features)

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("Model not trained")
        self.model.save_model(str(path))

    def load(self, path: Path) -> None:
        try:
            import lightgbm as lgb
        except OSError as exc:
            raise RuntimeError(
                "LightGBM failed to load. On macOS, install OpenMP with: brew install libomp"
            ) from exc

        self.model = lgb.Booster(model_file=str(path))


def _content_vector(row: pd.Series) -> np.ndarray:
    values = []
    for col in EMBEDDING_COLS:
        if col in row.index:
            val = row[col]
            values.append(0.0 if pd.isna(val) else float(val))
        else:
            values.append(0.0)
    return np.array(values, dtype=np.float32)


def build_ranking_features(
    candidates: list[str],
    retrieval_scores: dict[str, float],
    deep_scores: dict[str, float],
    item_features: pd.DataFrame,
    user_content_embedding: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    rows = []
    valid_ids = []
    user_emb = (
        np.zeros(len(EMBEDDING_COLS), dtype=np.float32)
        if user_content_embedding is None
        else user_content_embedding.astype(np.float32)
    )

    for item_id in candidates:
        feat = item_features[item_features["item_id"] == item_id]
        if feat.empty:
            continue
        row = feat.iloc[0]
        item_emb = _content_vector(row)
        content_sim = float(np.dot(user_emb, item_emb)) if user_emb.size else 0.0
        rows.append(
            [
                retrieval_scores.get(item_id, 0.0),
                deep_scores.get(item_id, 0.0),
                row.get("popularity_decay_7d", 0.0),
                row.get("popularity_decay_30d", 0.0),
                row.get("price_bucket", 0.0),
                row.get("category_id", 0.0),
                row.get("avg_rating", 0.0),
                row.get("link_degree", 0.0),
                row.get("avg_helpful_vote", 0.0),
                content_sim,
            ]
        )
        valid_ids.append(item_id)
    return np.array(rows, dtype=np.float32), valid_ids
