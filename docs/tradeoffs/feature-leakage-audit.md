# Causal Feature Leakage Audit

## Features Reviewed

| Feature | Contamination Risk | Mitigation |
|---------|-------------------|------------|
| `popularity_decay_7d` | High — shaped by prior recommendation policy | Time-capped aggregates; exclude post-policy-deployment data in training |
| `interaction_count_30d` | Medium — users shown more items accumulate more interactions | Propensity-weighted counts in training |
| `top_category_affinity` | Low — derived from organic interactions | No action needed |
| `price_bucket` | None — item metadata, not policy-dependent | No action needed |
| `retrieval_score` | High — direct model output from prior version | Use fresh retrieval model scores; avoid lagged scores as features |

## Recommendation

For production deployment, maintain a feature contamination registry updated each time the recommendation policy changes. Retrain features with explicit cutoff timestamps.
