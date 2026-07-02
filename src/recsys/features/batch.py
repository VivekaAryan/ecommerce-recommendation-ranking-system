"""Batch feature pipeline for training."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from recsys.config import get_base_config, load_yaml
from recsys.features.registry import FeatureRegistry
from recsys.utils import ensure_dir


def build_batch_features(
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    as_of: datetime | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_yaml("features.yaml")
    registry = FeatureRegistry(
        user_behavior_refresh_hours=cfg["staleness"]["user_behavior_refresh_hours"],
        item_popularity_refresh_hours=cfg["staleness"]["item_popularity_refresh_hours"],
        default_days_since_last=cfg["cold_start"]["default_days_since_last"],
        default_interaction_count=cfg["cold_start"]["default_interaction_count"],
        default_price_bucket=cfg["cold_start"]["default_price_bucket"],
    )
    as_of = as_of or datetime.utcfromtimestamp(int(interactions["timestamp"].max()))
    enriched = interactions.merge(items[["item_id", "price", "category"]], on="item_id", how="left")
    user_features = registry.compute_user_features(enriched, as_of, mode="batch")
    item_features = registry.compute_item_features(enriched, items, as_of, mode="batch")
    return user_features, item_features


def save_batch_features(
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    user_features, item_features = build_batch_features(interactions, items)
    ensure_dir(output_dir)
    paths = {
        "user_features": output_dir / "user_features_batch.parquet",
        "item_features": output_dir / "item_features_batch.parquet",
    }
    user_features.to_parquet(paths["user_features"], index=False)
    item_features.to_parquet(paths["item_features"], index=False)
    return paths


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = get_base_config()
    processed = cfg.resolve_path(cfg.paths.processed_dir)
    interactions = pd.read_parquet(processed / "interactions.parquet")
    items = pd.read_parquet(processed / "items.parquet")
    return interactions, items
