# GRU vs Transformer for Sequential Encoder

## Decision

Start with GRU for user history encoding in retrieval; SASRec-style transformer for ranking stage.

## Rationale

- GRU trains faster on subsampled data, enabling rapid Phase 1 iteration
- Ranking stage benefits more from self-attention over full candidate context
- Architectural story (sequential intent modeling) is preserved in both stages

## Upgrade Path

Replace `UserTower.gru` with a small transformer encoder (2 layers, 4 heads) when scaling to full dataset. Document expected recall improvement vs training time cost.
