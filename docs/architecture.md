# Architecture

## Funnel Stages

1. **Retrieval** — Two-tower model with GRU user encoder and content-aware item tower. Trained with logQ-corrected in-batch negatives. Served via FAISS ANN index (HNSW/IVF/flat benchmarks).
2. **Pre-rank** — Fast scoring over ~500 candidates using retrieval score + lightweight popularity/price features → top 100.
3. **Ranking** — SASRec-style sequential encoder + multi-task heads (click, purchase, expected value, engagement) combined with explicit weights. LightGBM LambdaRank hybrid head on side features.
4. **Re-ranking** — Isotonic calibration, MMR diversity, epsilon-greedy/Thompson exploration on tail slate positions.

## Dual Feature Pipeline

- **Batch path** — Features computed from full historical warehouse for training.
- **Online path** — Same feature definitions with injected staleness (user behavior refreshes every 6h, item popularity every 24h). Cold-start fallbacks for new users/items.

## Marketplace Simulator

Synthetic users with latent preference vectors interact with a real catalog that has limited inventory and attention decay (interference). Position bias drives probabilistic clicks/purchases. All impressions logged with propensities for off-policy evaluation.

## Evaluation

- Offline: NDCG, Recall, MAP, diversity, catalog coverage, Gini
- Counterfactual: IPS and doubly-robust OPE
- Interference: naive A/B vs switchback/cluster-randomized
- Long-run: multi-cycle feedback loop simulation

## Latency Budget

| Stage | Budget |
|-------|--------|
| Retrieval | 20ms |
| Pre-rank | 10ms |
| Ranking | 50ms |
| Re-ranking | 30ms |
| **Total** | **100ms** |
