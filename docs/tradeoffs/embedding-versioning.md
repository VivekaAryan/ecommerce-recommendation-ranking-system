# Embedding Versioning

## Decision

Anchor-based Procrustes alignment to maintain embedding space continuity across retrains.

## Rationale

Full FAISS reindex on every retrain is expensive at scale. Anchor items (fixed set of 1000 items) provide reference points to learn a rotation + translation aligning new embeddings to the old space.

## Implementation

1. Select anchor indices at first train
2. On retrain, compute Procrustes transform between old and new anchor embeddings
3. Apply transform to all new embeddings before index update

## Limitation

Alignment quality degrades if model architecture changes significantly. Major architecture changes require full reindex.
