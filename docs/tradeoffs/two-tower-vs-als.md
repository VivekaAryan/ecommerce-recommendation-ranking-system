# Two-Tower vs ALS

## Decision

Two-tower neural retrieval with sequence encoder and content-aware item tower.

## Rationale

- ALS/matrix factorization cannot incorporate item metadata or user sequence recency
- Two-tower scales to ANN serving (industry standard: YouTube, Pinterest)
- Sequence encoder captures intent shift without hand-engineered recency features

## Why Not ALS

ALS is simpler and faster to train but lacks cold-start content signals and sequential user modeling. For an interview artifact demonstrating production architecture, two-tower is the stronger choice.

## Cost

Training time and infrastructure complexity are higher. Mitigated by subsampled dataset and GRU (not full transformer) user encoder.
