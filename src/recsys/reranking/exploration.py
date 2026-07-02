"""Exploration strategies for slate positions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ExplorationDecision:
    item_id: str
    position: int
    explored: bool
    propensity: float


class EpsilonGreedyExplorer:
    def __init__(self, epsilon: float = 0.1, explore_positions: list[int] | None = None) -> None:
        self.epsilon = epsilon
        self.explore_positions = explore_positions or [8, 9]
        self.rng = np.random.default_rng(42)

    def apply(
        self,
        slate: list[str],
        catalog: list[str],
        relevance_scores: dict[str, float],
    ) -> list[ExplorationDecision]:
        decisions: list[ExplorationDecision] = []
        for position, item_id in enumerate(slate):
            explored = position in self.explore_positions and self.rng.random() < self.epsilon
            if explored:
                alternatives = [c for c in catalog if c not in slate]
                if alternatives:
                    item_id = self.rng.choice(alternatives)
            total = sum(relevance_scores.values()) or 1.0
            propensity = (1 - self.epsilon) * (relevance_scores.get(item_id, 0.0) / total) + self.epsilon * (
                1.0 / max(len(catalog), 1)
            )
            decisions.append(
                ExplorationDecision(item_id=item_id, position=position, explored=explored, propensity=propensity)
            )
        return decisions


class ThompsonSamplingExplorer:
    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        self.alpha = alpha
        self.beta = beta
        self.rng = np.random.default_rng(42)

    def sample_slate(
        self,
        candidates: list[str],
        success_counts: dict[str, int],
        trial_counts: dict[str, int],
        slate_size: int,
    ) -> list[ExplorationDecision]:
        sampled_scores = []
        for item_id in candidates:
            alpha = self.alpha + success_counts.get(item_id, 0)
            beta = self.beta + trial_counts.get(item_id, 0) - success_counts.get(item_id, 0)
            sampled_scores.append((item_id, self.rng.beta(alpha, beta)))
        sampled_scores.sort(key=lambda x: x[1], reverse=True)
        decisions = []
        total = sum(score for _, score in sampled_scores) or 1.0
        for position, (item_id, score) in enumerate(sampled_scores[:slate_size]):
            decisions.append(
                ExplorationDecision(
                    item_id=item_id,
                    position=position,
                    explored=True,
                    propensity=score / total,
                )
            )
        return decisions
