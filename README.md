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

# Download Amazon Reviews 2023 (7 categories, ~30-60 min)
python scripts/download_data.py

# Build features
python scripts/build_features.py

# Measure train/serve feature skew
python scripts/measure_skew.py

# Train retrieval
python scripts/train_retrieval.py

# Train ranker
python scripts/train_ranker.py

# Run simulator
python scripts/run_simulator.py

# Evaluate (offline + OPE)
python scripts/evaluate.py

# MLflow UI (after training)
mlflow ui --backend-store-uri sqlite:///data/mlflow.db
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

### macOS: Python quit unexpectedly

On Apple Silicon, PyTorch and FAISS can conflict over OpenMP and crash Python when you click **Get Recommendations**. The dashboard sets `KMP_DUPLICATE_LIB_OK=TRUE` automatically when started via `scripts/run_ui.py`. If you still see crashes, run:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
python scripts/run_ui.py --reload
```

LightGBM ranker training also needs Homebrew OpenMP on macOS:

```bash
brew install libomp
```

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

Amazon Reviews 2023 across **7 storefront categories** (Appliances, Books, Cell Phones and Accessories, Electronics, CDs and Vinyl, Musical Instruments, Toys and Games). The download samples **~175K interactions** (25K per category), keeps only products with **real Amazon `image_url` metadata**, and persists co-purchase / co-view links for retrieval boosting.

Expected disk usage after download + features + models: **~1–2 GB**. Download time is typically **30–60 minutes** depending on network speed.

```bash
python scripts/download_data.py
python scripts/build_features.py
python scripts/train_retrieval.py
python scripts/train_ranker.py
```

If download fails, the command exits with an error (no silent fallback). Ensure network access and optionally set `HF_TOKEN` for higher HuggingFace rate limits.

## Documentation

- [Architecture](docs/architecture.md)
- [Risk / Failure Modes](docs/risks.md)
- [Tradeoff Memos](docs/tradeoffs/)
- [Interview Narrative](docs/interview-narrative.md)

## License

MIT
