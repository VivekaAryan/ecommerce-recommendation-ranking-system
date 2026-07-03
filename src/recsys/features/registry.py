"""Canonical feature definitions shared by batch and online pipelines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd

from recsys.data.item_links import link_degree_by_item
from recsys.features.embeddings import EMBEDDING_COLS

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

        category_col = (
            "category"
            if "category" in df.columns
            else "main_category"
            if "main_category" in df.columns
            else "category"
        )
        grouped = df.groupby("user_id").agg(
            interaction_count_7d=("event_time", lambda s: (s >= as_of - timedelta(days=7)).sum()),
            interaction_count_30d=("event_time", lambda s: (s >= as_of - timedelta(days=30)).sum()),
            last_interaction=("event_time", "max"),
            top_category=(category_col, lambda s: s.mode().iloc[0] if len(s) else "Unknown"),
        )
        grouped["days_since_last_interaction"] = (
            (as_of - grouped["last_interaction"]).dt.total_seconds() / 86400
        ).fillna(self.default_days_since_last)
        grouped["top_category_affinity"] = grouped["top_category"].astype("category").cat.codes
        grouped["avg_price_interacted"] = df.groupby("user_id")["price"].mean().reindex(
            grouped.index
        ).fillna(0.0)

        if "verified_purchase" in df.columns:
            grouped["verified_purchase_rate"] = (
                df.groupby("user_id")["verified_purchase"].mean().reindex(grouped.index).fillna(0.0)
            )
        else:
            grouped["verified_purchase_rate"] = 0.0

        if "helpful_vote" in df.columns:
            grouped["avg_helpful_vote_authored"] = (
                df.groupby("user_id")["helpful_vote"].mean().reindex(grouped.index).fillna(0.0)
            )
        else:
            grouped["avg_helpful_vote_authored"] = 0.0

        return grouped.reset_index()

    def compute_item_features(
        self,
        interactions: pd.DataFrame,
        items: pd.DataFrame,
        as_of: datetime,
        mode: FeatureMode = "batch",
        item_links: pd.DataFrame | None = None,
        embedding_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        df = interactions.merge(
            items[[c for c in ["item_id", "price", "category", "main_category", "average_rating", "rating_number"] if c in items.columns]],
            on="item_id",
            how="left",
        )
        df["event_time"] = pd.to_datetime(df["timestamp"], unit="s")

        if mode == "online":
            cutoff = as_of - timedelta(hours=self.item_popularity_refresh_hours)
            df = df[df["event_time"] <= cutoff]

        grouped = df.groupby("item_id").agg(
            popularity_decay_7d=("event_time", lambda s: (s >= as_of - timedelta(days=7)).sum()),
            popularity_decay_30d=("event_time", lambda s: (s >= as_of - timedelta(days=30)).sum()),
            review_sentiment_proxy=("rating", "mean"),
        )

        category_col = "main_category" if "main_category" in items.columns else "category"
        item_cols = ["item_id", "price", category_col]
        for optional_col in ("average_rating", "rating_number"):
            if optional_col in items.columns:
                item_cols.append(optional_col)
        item_features = items[item_cols].copy()
        item_features = item_features.rename(columns={category_col: "category"})
        item_features["price_bucket"] = item_features["price"].map(self.price_to_bucket)
        item_features["category_id"] = item_features["category"].astype("category").cat.codes
        if "average_rating" in item_features.columns:
            item_features["avg_rating"] = item_features["average_rating"].fillna(
                item_features["item_id"].map(grouped["review_sentiment_proxy"])
            )
        else:
            item_features["avg_rating"] = item_features["item_id"].map(grouped["review_sentiment_proxy"]).fillna(0.0)
        if "rating_number" in item_features.columns:
            item_features["log_rating_number"] = item_features["rating_number"].fillna(0).map(
                lambda x: math.log1p(max(float(x), 0.0)) / 10.0
            )
        else:
            item_features["log_rating_number"] = 0.0

        degrees = link_degree_by_item(item_links) if item_links is not None else {}
        item_features["link_degree"] = item_features["item_id"].map(degrees).fillna(0).astype(int)

        if "helpful_vote" in interactions.columns:
            avg_helpful = (
                interactions.groupby("item_id")["helpful_vote"].mean()
            )
            item_features["avg_helpful_vote"] = (
                item_features["item_id"].map(avg_helpful).fillna(0.0)
            )
        else:
            item_features["avg_helpful_vote"] = 0.0

        item_features = item_features.merge(grouped, on="item_id", how="left")
        item_features[["popularity_decay_7d", "popularity_decay_30d"]] = item_features[
            ["popularity_decay_7d", "popularity_decay_30d"]
        ].fillna(0)
        item_features["review_sentiment_proxy"] = item_features["review_sentiment_proxy"].fillna(
            item_features["avg_rating"]
        )

        if embedding_df is not None:
            item_features = item_features.merge(embedding_df, on="item_id", how="left")
        else:
            for col in EMBEDDING_COLS:
                item_features[col] = 0.0

        drop_cols = ["price", "category", "average_rating", "rating_number"]
        return item_features.drop(columns=[c for c in drop_cols if c in item_features.columns])

    def cold_start_user_row(self, user_id: str) -> dict:
        return {
            "user_id": user_id,
            "interaction_count_7d": self.default_interaction_count,
            "interaction_count_30d": self.default_interaction_count,
            "days_since_last_interaction": self.default_days_since_last,
            "top_category_affinity": 0,
            "avg_price_interacted": 0.0,
            "verified_purchase_rate": 0.0,
            "avg_helpful_vote_authored": 0.0,
        }

    def cold_start_item_row(self, item: pd.Series) -> dict:
        row = {
            "item_id": item["item_id"],
            "price_bucket": self.price_to_bucket(item.get("price", np.nan)),
            "category_id": 0,
            "popularity_decay_7d": 0,
            "popularity_decay_30d": 0,
            "avg_rating": float(item.get("average_rating", 0.0) or 0.0),
            "log_rating_number": math.log1p(float(item.get("rating_number", 0) or 0.0)) / 10.0,
            "link_degree": 0,
            "avg_helpful_vote": 0.0,
            "review_sentiment_proxy": float(item.get("average_rating", 0.0) or 0.0),
        }
        for col in EMBEDDING_COLS:
            row[col] = 0.0
        return row
