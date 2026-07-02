"""SHAP-based interpretability for multi-task heads."""

from __future__ import annotations

import numpy as np
import pandas as pd


def explain_multitask_heads(
    model,
    features: pd.DataFrame,
    feature_cols: list[str],
    sample_size: int = 100,
) -> dict[str, pd.DataFrame]:
    try:
        import shap
    except ImportError as exc:
        raise ImportError("shap is required for interpretability") from exc

    sample = features[feature_cols].sample(n=min(sample_size, len(features)), random_state=42)
    explainer = shap.Explainer(model.predict, sample)
    shap_values = explainer(sample)
    return {
        "shap_values": pd.DataFrame(shap_values.values, columns=feature_cols),
        "base_value": float(np.mean(shap_values.base_values)),
    }


def head_disagreement_report(
    outputs: dict[str, np.ndarray],
    item_ids: list[str],
    top_k: int = 10,
) -> pd.DataFrame:
    rows = []
    for idx, item_id in enumerate(item_ids):
        rows.append(
            {
                "item_id": item_id,
                "click": float(outputs["click"][idx]),
                "purchase": float(outputs["purchase"][idx]),
                "expected_value": float(outputs["expected_value"][idx]),
                "engagement": float(outputs["engagement"][idx]),
            }
        )
    df = pd.DataFrame(rows)
    click_rank = df["click"].rank(ascending=False)
    value_rank = df["expected_value"].rank(ascending=False)
    df["rank_disagreement"] = (click_rank - value_rank).abs()
    return df.sort_values("rank_disagreement", ascending=False).head(top_k)
