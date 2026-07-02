"""Interference-aware experimentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ExperimentResult:
    design: str
    treatment_effect: float
    stderr: float


def naive_ab_test(logs: pd.DataFrame, metric_col: str = "clicked") -> ExperimentResult:
    treated = logs[logs["policy_id"].str.endswith("_treatment")]
    control = logs[~logs["policy_id"].str.endswith("_treatment")]
    effect = treated[metric_col].mean() - control[metric_col].mean()
    pooled = logs[metric_col].var()
    stderr = np.sqrt(pooled * (1 / max(len(treated), 1) + 1 / max(len(control), 1)))
    return ExperimentResult(design="naive_ab", treatment_effect=float(effect), stderr=float(stderr))


def switchback_test(
    logs: pd.DataFrame,
    time_col: str = "timestamp",
    metric_col: str = "clicked",
    window_hours: int = 6,
) -> ExperimentResult:
    df = logs.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)
    start = df[time_col].min()
    df["window"] = ((df[time_col] - start).dt.total_seconds() // (window_hours * 3600)).astype(int)
    effects = []
    for _window_id, group in df.groupby("window"):
        treated = group[group["policy_id"].str.endswith("_treatment")]
        control = group[~group["policy_id"].str.endswith("_treatment")]
        if len(treated) == 0 or len(control) == 0:
            continue
        effects.append(treated[metric_col].mean() - control[metric_col].mean())
    effect = float(np.mean(effects)) if effects else 0.0
    stderr = float(np.std(effects) / np.sqrt(len(effects))) if effects else 0.0
    return ExperimentResult(design="switchback", treatment_effect=effect, stderr=stderr)


def cluster_randomized_test(
    logs: pd.DataFrame,
    cluster_col: str,
    metric_col: str = "clicked",
) -> ExperimentResult:
    cluster_effects = []
    for _, group in logs.groupby(cluster_col):
        treated = group[group["policy_id"].str.endswith("_treatment")]
        control = group[~group["policy_id"].str.endswith("_treatment")]
        if len(treated) == 0 or len(control) == 0:
            continue
        cluster_effects.append(treated[metric_col].mean() - control[metric_col].mean())
    effect = float(np.mean(cluster_effects)) if cluster_effects else 0.0
    stderr = float(np.std(cluster_effects) / np.sqrt(len(cluster_effects))) if cluster_effects else 0.0
    return ExperimentResult(design="cluster_randomized", treatment_effect=effect, stderr=stderr)
