"""Canonical feature definitions shared by batch and online pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd

FeatureMode = Literal["batch", "online"]


@dataclass
class FeatureRegistry:
    user_behavior_refresh_hours: int = 6
    item_popularity_refresh_hours: int = 24
    default_days_since_last: int = 365
    default_interaction_count: int = 0
    default_price_bucket: int = 2

    def price_to_bucket(self, price: float) -> int:
        if np.isnan(price):
            return self.default_price_bucket
        if price < 25:
            return 0
        if price < 75:
            return 1
        if price < 200:
            return 2
        return 3

    def compute_user_features(
        self,
        interactions: pd.DataFrame,
        as_of: datetime,
        mode: FeatureMode = "batch",
    ) -> pd.DataFrame:
        df = interactions.copy()
        df["event_time"] = pd.to_datetime(df["timestamp"], unit="s")

        if mode == "online":
            cutoff = as_of - timedelta(hours=self.user_behavior_refresh_hours)
            df = df[df["event_time"] <= cutoff]

        grouped = df.groupby("user_id").agg(
            interaction_count_7d=("event_time", lambda s: (s >= as_of - timedelta(days=7)).sum()),
            interaction_count_30d=("event_time", lambda s: (s >= as_of - timedelta(days=30)).sum()),
            last_interaction=("event_time", "max"),
            top_category=("category", lambda s: s.mode().iloc[0] if len(s) else "Unknown"),
        )
        grouped["days_since_last_interaction"] = (
            (as_of - grouped["last_interaction"]).dt.total_seconds() / 86400
        ).fillna(self.default_days_since_last)
        grouped["top_category_affinity"] = grouped["top_category"].astype("category").cat.codes
        grouped["avg_price_interacted"] = df.groupby("user_id")["price"].mean().reindex(
            grouped.index
        ).fillna(0.0)
        return grouped.reset_index()

    def compute_item_features(
        self,
        interactions: pd.DataFrame,
        items: pd.DataFrame,
        as_of: datetime,
        mode: FeatureMode = "batch",
    ) -> pd.DataFrame:
        df = interactions.merge(items[["item_id", "price", "category"]], on="item_id", how="left")
        df["event_time"] = pd.to_datetime(df["timestamp"], unit="s")

        if mode == "online":
            cutoff = as_of - timedelta(hours=self.item_popularity_refresh_hours)
            df = df[df["event_time"] <= cutoff]

        grouped = df.groupby("item_id").agg(
            popularity_decay_7d=("event_time", lambda s: (s >= as_of - timedelta(days=7)).sum()),
            popularity_decay_30d=("event_time", lambda s: (s >= as_of - timedelta(days=30)).sum()),
        )
        item_features = items[["item_id", "price", "category"]].copy()
        item_features["price_bucket"] = item_features["price"].map(self.price_to_bucket)
        item_features["category_id"] = item_features["category"].astype("category").cat.codes
        item_features = item_features.merge(grouped, on="item_id", how="left")
        item_features[["popularity_decay_7d", "popularity_decay_30d"]] = item_features[
            ["popularity_decay_7d", "popularity_decay_30d"]
        ].fillna(0)
        return item_features.drop(columns=["price", "category"])

    def cold_start_user_row(self, user_id: str) -> dict:
        return {
            "user_id": user_id,
            "interaction_count_7d": self.default_interaction_count,
            "interaction_count_30d": self.default_interaction_count,
            "days_since_last_interaction": self.default_days_since_last,
            "top_category_affinity": 0,
            "avg_price_interacted": 0.0,
        }

    def cold_start_item_row(self, item: pd.Series) -> dict:
        return {
            "item_id": item["item_id"],
            "price_bucket": self.price_to_bucket(item.get("price", np.nan)),
            "category_id": 0,
            "popularity_decay_7d": 0,
            "popularity_decay_30d": 0,
        }
