"""Batch feature pipeline for training."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from recsys.config import get_base_config, load_yaml
from recsys.data.item_links import load_item_links
from recsys.data.timestamps import normalize_timestamps
from recsys.features.embeddings import compute_content_embeddings
from recsys.features.registry import FeatureRegistry
from recsys.utils import ensure_dir


def build_batch_features(
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    as_of: datetime | None = None,
    features_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_yaml("features.yaml")
    registry = FeatureRegistry(
        user_behavior_refresh_hours=cfg["staleness"]["user_behavior_refresh_hours"],
        item_popularity_refresh_hours=cfg["staleness"]["item_popularity_refresh_hours"],
        default_days_since_last=cfg["cold_start"]["default_days_since_last"],
        default_interaction_count=cfg["cold_start"]["default_interaction_count"],
        default_price_bucket=cfg["cold_start"]["default_price_bucket"],
    )
    interactions = normalize_timestamps(interactions)
    as_of = as_of or datetime.utcfromtimestamp(int(interactions["timestamp"].max()))

    base_cfg = get_base_config()
    processed_dir = base_cfg.resolve_path(base_cfg.paths.processed_dir)
    item_links = load_item_links(processed_dir)

    embedding_df, _ = compute_content_embeddings(items, reviews=interactions, output_dir=features_dir)

    category_col = "main_category" if "main_category" in items.columns else "category"
    item_cols = ["item_id", "price", category_col]
    interaction_cols = [
        c
        for c in interactions.columns
        if c not in {category_col, "category", "price"}
    ]
    enriched = interactions[interaction_cols].merge(items[item_cols], on="item_id", how="left")
    if category_col != "category":
        enriched = enriched.rename(columns={category_col: "category"})

    user_features = registry.compute_user_features(enriched, as_of, mode="batch")
    item_features = registry.compute_item_features(
        enriched,
        items,
        as_of,
        mode="batch",
        item_links=item_links,
        embedding_df=embedding_df,
    )
    return user_features, item_features


def save_batch_features(
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    ensure_dir(output_dir)
    user_features, item_features = build_batch_features(interactions, items, features_dir=output_dir)
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
    interactions = normalize_timestamps(pd.read_parquet(processed / "interactions.parquet"))
    items = pd.read_parquet(processed / "items.parquet")
    return interactions, items
