"""Adversarial gaming detection and mitigation."""

from __future__ import annotations

import pandas as pd


def inject_synthetic_manipulation(
    logs: pd.DataFrame,
    target_items: list[str],
    boost_factor: float = 5.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Inject artificial engagement spikes for selected items."""
    df = logs.copy()
    mask = df["item_id"].isin(target_items)
    boosted = df[mask].sample(frac=min(1.0, boost_factor / 10), replace=True, random_state=seed)
    boosted["clicked"] = True
    if "purchased" in boosted.columns:
        boosted["purchased"] = boosted["purchased"] | True
    else:
        boosted["purchased"] = True
    return pd.concat([df, boosted], ignore_index=True)


def detect_engagement_velocity_anomalies(
    logs: pd.DataFrame,
    window: str = "1D",
    z_threshold: float = 3.0,
) -> pd.DataFrame:
    df = logs.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    daily = (
        df.groupby(["item_id", pd.Grouper(key="timestamp", freq=window)])
        .agg(clicks=("clicked", "sum"), impressions=("item_id", "count"))
        .reset_index()
    )
    daily["ctr"] = daily["clicks"] / daily["impressions"].clip(lower=1)
    mean_ctr = daily["ctr"].mean()
    std_ctr = daily["ctr"].std() or 1e-6
    daily["z_score"] = (daily["ctr"] - mean_ctr) / std_ctr
    return daily[daily["z_score"] > z_threshold]
