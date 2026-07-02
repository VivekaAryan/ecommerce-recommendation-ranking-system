# Production-Grade Recommendation & Ranking Platform

A multi-stage ecommerce recommendation system built to mirror production personalization stacks: retrieval → pre-rank → ranking → re-ranking, with train/serve skew measurement, counterfactual evaluation, and a marketplace simulator.

## Architecture

```
[Full catalog]
     → Retrieval (two-tower + FAISS ANN)
     → Pre-rank (~500 → ~100)
     → Ranking (sequential multi-task + LightGBM)
     → Re-ranking (calibration, MMR, exploration)
     → Slate + propensity logging → feedback loop
```

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Prepare synthetic data (local dev)
python scripts/download_data.py --synthetic --size 10000

# Build features
python scripts/build_features.py --synthetic

# Measure train/serve feature skew
python scripts/measure_skew.py

# Train retrieval
python scripts/train_retrieval.py --synthetic

# Train ranker
python scripts/train_ranker.py --synthetic

# Run simulator
python scripts/run_simulator.py --synthetic

# Evaluate (offline + OPE)
python scripts/evaluate.py --synthetic

# MLflow UI
mlflow ui --backend-store-uri mlruns
```

## Testing Dashboard

A web UI for running the pipeline, testing live recommendations, and inspecting metrics.

### Development (hot reload)

Terminal 1 — API server:
```bash
source .venv/bin/activate
pip install -e ".[dev,serving]"
python scripts/run_ui.py --reload
```

Terminal 2 — Frontend dev server:
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

### Production (single server)

```bash
cd frontend && npm install && npm run build && cd ..
python scripts/run_ui.py
```

Open http://localhost:8000

### Dashboard Features

- **Overview** — artifact status, dataset stats, one-click full pipeline
- **Pipeline** — run each stage individually (data, features, skew, retrieval, ranker, simulator, evaluate)
- **Recommend** — pick a user and inspect the full funnel slate with latency breakdown
- **Metrics** — OPE estimates, simulator summary, feature skew table
- **Simulator** — browse propensity-logged impressions

## Phase Status

| Phase | Component | Status |
|-------|-----------|--------|
| 0 | Data ingestion, dual feature pipeline, simulator, MLflow | Implemented |
| 1 | Two-tower retrieval, FAISS ANN, cold-start, embedding versioning | Implemented |
| 2 | Sequential + multi-task ranking, LightGBM hybrid | Implemented |
| 3 | Calibration, MMR diversity, exploration, latency profiling | Implemented |
| 4 | Offline metrics, IPS/DR OPE, interference, feedback loop | Implemented |
| 5 | Adversarial gaming, SHAP interpretability, interview narrative | Implemented |

## Project Layout

- `src/recsys/` — core library
- `configs/` — YAML experiment configuration
- `scripts/` — CLI entrypoints
- `docs/` — architecture, risks, tradeoff memos
- `tests/` — unit tests

## Dataset

Amazon Reviews 2023 (Electronics), subsampled to ~500K–1M interactions for local iteration. Use `--synthetic` for fast local development without downloading the full dataset.

## Documentation

- [Architecture](docs/architecture.md)
- [Risk / Failure Modes](docs/risks.md)
- [Tradeoff Memos](docs/tradeoffs/)
- [Interview Narrative](docs/interview-narrative.md)

## License

MIT
