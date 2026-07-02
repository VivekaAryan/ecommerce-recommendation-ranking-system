# Interview Narrative

## Problem

Build a production-grade ecommerce recommendation system that demonstrates systems thinking beyond "train a model, report NDCG" — covering train/serve skew, counterfactual evaluation, interference, feedback loops, and adversarial robustness.

## Architecture (30 seconds)

Four-stage funnel: two-tower retrieval with FAISS → cheap pre-rank → sequential multi-task ranker with LightGBM hybrid head → calibrated, diverse, exploration-aware re-ranking. Dual feature pipelines (batch + stale online) from day one. Marketplace simulator with inventory interference and propensity logging as the evaluation spine.

## Three Hardest Tradeoffs

1. **Subsampled vs full Electronics** — Chose 500K–1M interactions for local GPU iteration with a documented scale-up path. Tradeoff: less tail-item signal, but full pipeline runnable on a laptop.

2. **GRU vs Transformer for user encoder** — Started with GRU for faster training iteration. Documented upgrade path to transformer. The architectural story (sequence encoding for intent) matters more than the specific encoder at portfolio scale.

3. **Multi-task head weighting** — Click (0.35), purchase (0.30), expected value (0.25), engagement (0.10). This is a product decision: prioritize conversion while keeping a value head to catch clickbait the click head would surface.

## What I'd Do at 10x Scale

- Shard catalog by category/region with per-shard ANN indexes
- Move feature serving to a real feature store (Feast/Tecton) with SLA monitoring
- Replace local MLflow with managed experiment tracking + model registry
- Add real-time embedding updates via streaming retrain pipeline
- Deploy switchback experiments in production with automated guardrails

## Key Artifacts

- Working code across all 4 funnel stages
- Recall/latency ANN benchmark curves
- IPS vs doubly-robust OPE comparison
- Naive A/B vs switchback interference demo
- Feedback loop coverage collapse measurement
- Risk doc with mitigations for 10 failure modes
