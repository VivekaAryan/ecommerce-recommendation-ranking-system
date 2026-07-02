"""Anchor-based embedding alignment across retrains."""

from __future__ import annotations

import numpy as np


def select_anchor_indices(num_items: int, num_anchors: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    num_anchors = min(num_anchors, num_items)
    return np.sort(rng.choice(num_items, size=num_anchors, replace=False))


def procrustes_align(old_anchors: np.ndarray, new_anchors: np.ndarray) -> np.ndarray:
    """Learn rotation matrix aligning new embedding space to old via anchor items."""
    old_mean = old_anchors.mean(axis=0)
    new_mean = new_anchors.mean(axis=0)
    old_centered = old_anchors - old_mean
    new_centered = new_anchors - new_mean
    cross = new_centered.T @ old_centered
    u, _, vt = np.linalg.svd(cross)
    rotation = u @ vt
    translation = old_mean - new_mean @ rotation
    return np.vstack([rotation, translation])


def apply_alignment(embeddings: np.ndarray, transform: np.ndarray) -> np.ndarray:
    rotation = transform[:-1]
    translation = transform[-1]
    return embeddings @ rotation + translation
