"""Offline ranking metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def dcg_at_k(relevances: list[float], k: int) -> float:
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(predicted: list[str], ground_truth: set[str], k: int) -> float:
    relevances = [1.0 if item in ground_truth else 0.0 for item in predicted[:k]]
    ideal = sorted(relevances, reverse=True)
    denom = dcg_at_k(ideal, k)
    if denom == 0:
        return 0.0
    return dcg_at_k(relevances, k) / denom


def recall_at_k(predicted: list[str], ground_truth: set[str], k: int) -> float:
    if not ground_truth:
        return 0.0
    return len(set(predicted[:k]) & ground_truth) / len(ground_truth)


def average_precision(predicted: list[str], ground_truth: set[str]) -> float:
    if not ground_truth:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, item in enumerate(predicted, start=1):
        if item in ground_truth:
            hits += 1
            precision_sum += hits / i
    return precision_sum / len(ground_truth)


def catalog_coverage(recommendations: pd.DataFrame, catalog_size: int) -> float:
    unique = recommendations["item_id"].nunique()
    return unique / catalog_size if catalog_size else 0.0


def gini_coefficient(counts: np.ndarray) -> float:
    if len(counts) == 0:
        return 0.0
    sorted_counts = np.sort(counts)
    n = len(counts)
    cumulative = np.cumsum(sorted_counts)
    return (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n


def evaluate_recommendations(
    predictions: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    k: int = 10,
) -> dict[str, float]:
    ndcgs, recalls, maps = [], [], []
    for user_id, preds in predictions.items():
        truth = ground_truth.get(user_id, set())
        ndcgs.append(ndcg_at_k(preds, truth, k))
        recalls.append(recall_at_k(preds, truth, k))
        maps.append(average_precision(preds, truth))
    return {
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        "map": float(np.mean(maps)) if maps else 0.0,
    }
