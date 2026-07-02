# Risk & Failure Mode Document

Design-review style catalog of known failure modes and mitigations.

## 1. Stale ANN Index

**Risk:** Item embeddings retrained but FAISS index not rebuilt → retrieval serves outdated neighbors.

**Mitigation:** Anchor-based embedding alignment (`embedding_versioning.py`) allows incremental index updates. Version tag on index artifacts; serving checks index/model version match.

## 2. Feature Pipeline Outage

**Risk:** Online feature store unavailable → serving falls back to defaults, degrading ranking quality silently.

**Mitigation:** Dual pipeline with explicit cold-start defaults. Skew measurement script alerts when batch vs online distributions diverge beyond threshold. Fallback to content-only retrieval for new users.

## 3. Cold-Start Users/Items

**Risk:** No interaction history → collaborative signals absent, poor recommendations.

**Mitigation:** Content-based cold-start path in retrieval (`cold_start.py`). Item tower produces embeddings from metadata immediately. Category-affinity fallback slates.

## 4. Feedback Loop / Catalog Collapse

**Risk:** Recommender reinforces popular items → training data becomes increasingly biased → coverage collapses.

**Mitigation:** Exploration budget (epsilon-greedy on tail positions). Popularity debiasing in training (logQ correction). Feedback loop simulation (`run_feedback_loop.py`) monitors Gini and catalog coverage over retrain cycles.

## 5. Uncalibrated Scores

**Risk:** Raw model scores used as probabilities → exploration and business rules mis-calibrated.

**Mitigation:** Isotonic regression post-ranking. Reliability diagrams per experiment. Calibration checked before deploying exploration.

## 6. Adversarial Gaming

**Risk:** Synthetic engagement manipulation inflates item rankings.

**Mitigation:** Engagement velocity anomaly detection (`adversarial.py`). Flag items with CTR z-score > 3. Down-weight suspicious items in re-ranking.

## 7. Interference in A/B Tests

**Risk:** Shared inventory means naive A/B tests overestimate treatment effect.

**Mitigation:** Switchback and cluster-randomized experiment designs (`interference.py`). Document bias in naive estimates.

## 8. Train/Serve Skew

**Risk:** Training features fresher than serving features → silent quality regression at deploy.

**Mitigation:** Built both pipelines from day one. `measure_skew.py` quantifies divergence. Ranker evaluated with both feature sets.

## 9. Latency Budget Exceeded

**Risk:** Full funnel exceeds 100ms under load.

**Mitigation:** Per-stage profiling in `ServingPipeline`. Documented degradation: reduce candidates (500→200), skip exploration, use IVF instead of exact search.

## 10. Causal Feature Leakage

**Risk:** Popularity features contaminated by prior model decisions.

**Mitigation:** Feature leakage audit (`docs/tradeoffs/feature-leakage-audit.md`). Time-capped aggregates, propensity-weighted features.
