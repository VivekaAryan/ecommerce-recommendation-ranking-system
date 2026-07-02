"""Multi-task ranking heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MultiTaskWeights:
    click: float = 0.35
    purchase: float = 0.30
    expected_value: float = 0.25
    engagement: float = 0.10

    def as_dict(self) -> dict[str, float]:
        return {
            "click": self.click,
            "purchase": self.purchase,
            "expected_value": self.expected_value,
            "engagement": self.engagement,
        }


class MultiTaskRanker(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.click_head = nn.Linear(hidden_dim, 1)
        self.purchase_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.engagement_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.shared(features)
        return {
            "click": torch.sigmoid(self.click_head(shared)).squeeze(-1),
            "purchase": torch.sigmoid(self.purchase_head(shared)).squeeze(-1),
            "expected_value": self.value_head(shared).squeeze(-1),
            "engagement": torch.sigmoid(self.engagement_head(shared)).squeeze(-1),
        }

    def combined_score(self, outputs: dict[str, torch.Tensor], weights: MultiTaskWeights) -> torch.Tensor:
        w = weights.as_dict()
        return (
            w["click"] * outputs["click"]
            + w["purchase"] * outputs["purchase"]
            + w["expected_value"] * torch.sigmoid(outputs["expected_value"])
            + w["engagement"] * outputs["engagement"]
        )
