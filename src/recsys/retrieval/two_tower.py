"""Two-tower retrieval model with logQ-corrected in-batch negatives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TwoTowerConfig:
    num_users: int
    num_items: int
    embedding_dim: int = 128
    user_hidden_dim: int = 256
    item_hidden_dim: int = 256
    history_length: int = 50
    item_feature_dim: int = 16
    dropout: float = 0.1
    temperature: float = 0.05


class UserTower(nn.Module):
    def __init__(self, cfg: TwoTowerConfig) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(cfg.num_items, cfg.embedding_dim, padding_idx=0)
        self.gru = nn.GRU(cfg.embedding_dim, cfg.user_hidden_dim, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(cfg.user_hidden_dim, cfg.embedding_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, history_item_ids: torch.Tensor) -> torch.Tensor:
        emb = self.item_embedding(history_item_ids)
        _, hidden = self.gru(emb)
        return F.normalize(self.proj(hidden.squeeze(0)), dim=-1)


class ItemTower(nn.Module):
    def __init__(self, cfg: TwoTowerConfig) -> None:
        super().__init__()
        self.id_embedding = nn.Embedding(cfg.num_items, cfg.embedding_dim)
        self.feature_proj = nn.Sequential(
            nn.Linear(cfg.item_feature_dim, cfg.item_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.item_hidden_dim, cfg.embedding_dim),
        )
        self.combine = nn.Linear(cfg.embedding_dim * 2, cfg.embedding_dim)

    def forward(self, item_ids: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        id_emb = self.id_embedding(item_ids)
        feat_emb = self.feature_proj(item_features)
        combined = self.combine(torch.cat([id_emb, feat_emb], dim=-1))
        return F.normalize(combined, dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, cfg: TwoTowerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.user_tower = UserTower(cfg)
        self.item_tower = ItemTower(cfg)

    def encode_users(self, history_item_ids: torch.Tensor) -> torch.Tensor:
        return self.user_tower(history_item_ids)

    def encode_items(self, item_ids: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        return self.item_tower(item_ids, item_features)

    def forward(
        self,
        history_item_ids: torch.Tensor,
        item_ids: torch.Tensor,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        user_emb = self.encode_users(history_item_ids)
        item_emb = self.encode_items(item_ids, item_features)
        return (user_emb * item_emb).sum(dim=-1)


def logq_corrected_loss(
    logits: torch.Tensor,
    item_popularity: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """In-batch negative sampling with logQ correction for popularity bias."""
    corrected = logits - torch.log(item_popularity.unsqueeze(0) + 1e-8)
    corrected = corrected / temperature
    labels = torch.arange(corrected.size(0), device=corrected.device)
    return F.cross_entropy(corrected, labels)


def compute_inbatch_logits(user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
    return user_emb @ item_emb.T
