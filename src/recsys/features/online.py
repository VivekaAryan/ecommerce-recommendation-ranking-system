"""Online feature pipeline with staleness injection."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from recsys.config import load_yaml
from recsys.features.registry import FeatureRegistry


class OnlineFeaturePipeline:
    def __init__(self, registry: FeatureRegistry | None = None) -> None:
        cfg = load_yaml("features.yaml")
        self.registry = registry or FeatureRegistry(
            user_behavior_refresh_hours=cfg["staleness"]["user_behavior_refresh_hours"],
            item_popularity_refresh_hours=cfg["staleness"]["item_popularity_refresh_hours"],
            default_days_since_last=cfg["cold_start"]["default_days_since_last"],
            default_interaction_count=cfg["cold_start"]["default_interaction_count"],
            default_price_bucket=cfg["cold_start"]["default_price_bucket"],
        )

    def get_user_features(
        self,
        user_id: str,
        interactions: pd.DataFrame,
        items: pd.DataFrame,
        as_of: datetime,
    ) -> pd.Series:
        user_history = interactions[interactions["user_id"] == user_id]
        if user_history.empty:
            return pd.Series(self.registry.cold_start_user_row(user_id))

        enriched = user_history.merge(items[["item_id", "price", "category"]], on="item_id", how="left")
        features = self.registry.compute_user_features(enriched, as_of, mode="online")
        row = features[features["user_id"] == user_id]
        if row.empty:
            return pd.Series(self.registry.cold_start_user_row(user_id))
        return row.iloc[0]

    def get_item_features(
        self,
        item_id: str,
        interactions: pd.DataFrame,
        items: pd.DataFrame,
        as_of: datetime,
    ) -> pd.Series:
        item_row = items[items["item_id"] == item_id]
        if item_row.empty:
            raise KeyError(f"Unknown item_id: {item_id}")

        item_history = interactions[interactions["item_id"] == item_id]
        if item_history.empty:
            return pd.Series(self.registry.cold_start_item_row(item_row.iloc[0]))

        features = self.registry.compute_item_features(item_history, items, as_of, mode="online")
        row = features[features["item_id"] == item_id]
        if row.empty:
            return pd.Series(self.registry.cold_start_item_row(item_row.iloc[0]))
        return row.iloc[0]

    def build_online_snapshot(
        self,
        interactions: pd.DataFrame,
        items: pd.DataFrame,
        as_of: datetime | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        as_of = as_of or datetime.utcnow()
        enriched = interactions.merge(items[["item_id", "price", "category"]], on="item_id", how="left")
        user_features = self.registry.compute_user_features(enriched, as_of, mode="online")
        item_features = self.registry.compute_item_features(enriched, items, as_of, mode="online")
        return user_features, item_features


def measure_feature_skew(
    batch_user: pd.DataFrame,
    online_user: pd.DataFrame,
    numeric_cols: list[str] | None = None,
) -> pd.DataFrame:
    numeric_cols = numeric_cols or [
        "interaction_count_7d",
        "interaction_count_30d",
        "days_since_last_interaction",
        "avg_price_interacted",
    ]
    rows = []
    for col in numeric_cols:
        if col not in batch_user.columns or col not in online_user.columns:
            continue
        batch_mean = batch_user[col].mean()
        online_mean = online_user[col].mean()
        delta = online_mean - batch_mean
        rel = delta / batch_mean if batch_mean else 0.0
        rows.append(
            {
                "feature": col,
                "batch_mean": batch_mean,
                "online_mean": online_mean,
                "absolute_delta": delta,
                "relative_delta": rel,
            }
        )
    return pd.DataFrame(rows)
