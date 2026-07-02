"""Off-policy evaluation: IPS and doubly robust estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def ips_estimator(
    logs: pd.DataFrame,
    reward_col: str = "clicked",
    propensity_col: str = "propensity",
    new_propensity_col: str | None = None,
    clip: float = 100.0,
) -> float:
    rewards = logs[reward_col].astype(float)
    old_propensity = logs[propensity_col].clip(lower=1e-6)
    if new_propensity_col and new_propensity_col in logs.columns:
        new_propensity = logs[new_propensity_col].clip(lower=1e-6)
    else:
        new_propensity = np.ones(len(logs))
    weights = (new_propensity / old_propensity).clip(upper=clip)
    return float((weights * rewards).mean())


def fit_reward_model(logs: pd.DataFrame, feature_cols: list[str], reward_col: str = "clicked"):
    model = LogisticRegression(max_iter=1000)
    model.fit(logs[feature_cols], logs[reward_col])
    return model


def doubly_robust_estimator(
    logs: pd.DataFrame,
    reward_model: LogisticRegression,
    feature_cols: list[str],
    reward_col: str = "clicked",
    propensity_col: str = "propensity",
    new_propensity_col: str | None = None,
    clip: float = 100.0,
) -> float:
    rewards = logs[reward_col].astype(float).to_numpy()
    old_propensity = logs[propensity_col].clip(lower=1e-6).to_numpy()
    if new_propensity_col and new_propensity_col in logs.columns:
        new_propensity = logs[new_propensity_col].clip(lower=1e-6).to_numpy()
    else:
        new_propensity = np.ones(len(logs))
    q_hat = reward_model.predict_proba(logs[feature_cols])[:, 1]
    weights = (new_propensity / old_propensity).clip(max=clip)
    dr = q_hat + weights * (rewards - q_hat)
    return float(dr.mean())


def compare_ope_estimators(logs: pd.DataFrame, feature_cols: list[str]) -> dict[str, float]:
    ips = ips_estimator(logs)
    model = fit_reward_model(logs, feature_cols)
    dr = doubly_robust_estimator(logs, model, feature_cols)
    naive = float(logs["clicked"].mean())
    return {"naive": naive, "ips": ips, "doubly_robust": dr}
