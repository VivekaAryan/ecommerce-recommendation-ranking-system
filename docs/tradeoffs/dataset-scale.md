# Dataset Scale: Subsampled vs Full Electronics

## Decision

Subsample Amazon Reviews 2023 Electronics to ~500K–1M interactions for local development, keeping all items referenced by sampled interactions.

## Rationale

- Full Electronics category is multi-GB and requires significant download/streaming time
- Local GPU (MPS/CUDA) can train two-tower + sequential models on subsample in reasonable time
- Temporal distribution preserved via stratified time-bucket sampling

## Tradeoffs

| Subsampled | Full |
|---|---|
| Fast iteration | Realistic tail-item distribution |
| Runs on laptop GPU | Better retrieval recall on rare items |
| Good for pipeline validation | Required for production-grade metrics |

## Scale-up Path

1. Increase `target_interactions` in `configs/base.yaml`
2. Run `scripts/download_data.py` without `--synthetic`
3. Re-tune ANN index (IVF nlist/nprobe) for larger catalog
4. Consider distributed training for sequential ranker
