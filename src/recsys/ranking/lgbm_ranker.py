"""LightGBM LambdaRank hybrid head."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


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
        import lightgbm as lgb

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
        import lightgbm as lgb

        self.model = lgb.Booster(model_file=str(path))


def build_ranking_features(
    candidates: list[str],
    retrieval_scores: dict[str, float],
    deep_scores: dict[str, float],
    item_features: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    rows = []
    valid_ids = []
    for item_id in candidates:
        feat = item_features[item_features["item_id"] == item_id]
        if feat.empty:
            continue
        row = feat.iloc[0]
        rows.append(
            [
                retrieval_scores.get(item_id, 0.0),
                deep_scores.get(item_id, 0.0),
                row.get("popularity_decay_7d", 0.0),
                row.get("popularity_decay_30d", 0.0),
                row.get("price_bucket", 0.0),
                row.get("category_id", 0.0),
            ]
        )
        valid_ids.append(item_id)
    return np.array(rows, dtype=np.float32), valid_ids
