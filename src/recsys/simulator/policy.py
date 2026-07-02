"""Slate serving policy with position bias and probabilistic interactions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recsys.simulator.environment import MarketplaceEnvironment


@dataclass
class InteractionOutcome:
    user_id: str
    item_id: str
    position: int
    relevance: float
    score: float
    propensity: float
    clicked: bool
    purchased: bool
    policy_id: str
    explored: bool


class ServingPolicy:
    def __init__(
        self,
        position_bias: list[float],
        click_temperature: float = 1.0,
        purchase_given_click_prob: float = 0.15,
        policy_id: str = "baseline_v0",
        rng: np.random.Generator | None = None,
    ) -> None:
        self.position_bias = position_bias
        self.click_temperature = click_temperature
        self.purchase_given_click_prob = purchase_given_click_prob
        self.policy_id = policy_id
        self.rng = rng or np.random.default_rng(42)

    def score_candidates(
        self,
        env: MarketplaceEnvironment,
        user_id: str,
        candidate_ids: list[str],
    ) -> list[tuple[str, float]]:
        scored = [(item_id, env.relevance(user_id, item_id)) for item_id in candidate_ids]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def build_slate(
        self,
        env: MarketplaceEnvironment,
        user_id: str,
        candidate_ids: list[str],
        slate_size: int,
    ) -> list[tuple[str, float, float]]:
        scored = self.score_candidates(env, user_id, candidate_ids)[:slate_size]
        total = sum(score for _, score in scored) or 1.0
        slate = []
        for item_id, score in scored:
            propensity = score / total
            slate.append((item_id, score, propensity))
        return slate

    def simulate_interactions(
        self,
        env: MarketplaceEnvironment,
        user_id: str,
        slate: list[tuple[str, float, float]],
    ) -> list[InteractionOutcome]:
        outcomes: list[InteractionOutcome] = []
        for position, (item_id, score, propensity) in enumerate(slate):
            relevance = env.relevance(user_id, item_id)
            position_weight = self.position_bias[min(position, len(self.position_bias) - 1)]
            click_prob = min(1.0, relevance * position_weight / self.click_temperature)
            clicked = self.rng.random() < click_prob
            purchased = clicked and self.rng.random() < self.purchase_given_click_prob
            if clicked:
                env.serve_item(item_id)
            outcomes.append(
                InteractionOutcome(
                    user_id=user_id,
                    item_id=item_id,
                    position=position,
                    relevance=relevance,
                    score=score,
                    propensity=propensity,
                    clicked=clicked,
                    purchased=purchased,
                    policy_id=self.policy_id,
                    explored=False,
                )
            )
        return outcomes
