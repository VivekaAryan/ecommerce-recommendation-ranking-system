# Latency Budget

## Budget

| Stage | Target | What Gets Cut Under Pressure |
|-------|--------|------------------------------|
| Retrieval | 20ms | Reduce ANN top-K (500→200); use IVF instead of HNSW |
| Pre-rank | 10ms | Skip pre-rank; pass top-200 directly to ranker |
| Ranking | 50ms | Use LightGBM only (skip deep model forward pass) |
| Re-ranking | 30ms | Skip exploration; reduce MMR candidate pool |
| **Total** | **100ms** | |

## Profiling

`ServingPipeline` measures per-stage latency on every request. `within_budget()` returns pass/fail per stage.

## Segment-Based Degradation

For low-value user segments (new users, low engagement), serve simplified ranker with content-only retrieval to stay within budget.
