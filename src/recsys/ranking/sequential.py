"""SASRec-style sequential ranker."""

from __future__ import annotations

import torch
import torch.nn as nn


class SequentialRanker(nn.Module):
    def __init__(
        self,
        num_items: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        max_seq_length: int = 50,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_length, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.max_seq_length = max_seq_length

    def forward(self, item_seq: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = item_seq.shape
        positions = torch.arange(seq_len, device=item_seq.device).unsqueeze(0).expand(batch_size, -1)
        x = self.item_embedding(item_seq) + self.pos_embedding(positions)
        mask = self._causal_mask(seq_len, item_seq.device)
        encoded = self.encoder(x, mask=mask)
        return encoded[:, -1, :]

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device) * float("-inf"), diagonal=1)

    def score_candidates(self, item_seq: torch.Tensor, candidate_emb: torch.Tensor) -> torch.Tensor:
        user_repr = self.forward(item_seq)
        return torch.matmul(user_repr, candidate_emb.T)
