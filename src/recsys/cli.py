"""Command-line entrypoints."""

from __future__ import annotations

from recsys.env import configure_runtime_env

configure_runtime_env()

import argparse
from pathlib import Path

from recsys.config import get_base_config
from recsys.data.download import prepare_dataset
from recsys.data.splits import load_splits
from recsys.features.batch import load_processed_data, save_batch_features
from recsys.features.online import OnlineFeaturePipeline, measure_feature_skew
from recsys.simulator.runner import build_default_runner
from recsys.utils import ensure_dir


def _require_processed_data() -> Path:
    cfg = get_base_config()
    processed = cfg.resolve_path(cfg.paths.processed_dir)
    interactions_path = processed / "interactions.parquet"
    if not interactions_path.exists():
        raise FileNotFoundError(
            "Dataset not prepared. Run: python scripts/download_data.py"
        )
    return processed


def download_data() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare Amazon Reviews 2023 data")
    parser.parse_args()
    paths = prepare_dataset()
    print(f"Prepared dataset: {paths}")


def build_features() -> None:
    parser = argparse.ArgumentParser(description="Build batch feature tables")
    parser.parse_args()
    _require_processed_data()
    interactions, items = load_processed_data()
    cfg = get_base_config()
    paths = save_batch_features(interactions, items, cfg.resolve_path(cfg.paths.features_dir))
    print(f"Saved features: {paths}")


def train_retrieval() -> None:
    parser = argparse.ArgumentParser(description="Train two-tower retrieval model")
    parser.parse_args()
    cfg = get_base_config()
    processed = _require_processed_data()
    features_dir = cfg.resolve_path(cfg.paths.features_dir)
    _, items, train, val, _ = load_splits(processed)
    item_features = None
    item_features_path = features_dir / "item_features_batch.parquet"
    if item_features_path.exists():
        item_features = __import__("pandas").read_parquet(item_features_path)
    from recsys.retrieval.trainer import train_retrieval_model

    train_retrieval_model(
        train,
        val,
        items,
        cfg.resolve_path(cfg.paths.models_dir) / "retrieval",
        item_features=item_features,
    )


def train_ranker() -> None:
    parser = argparse.ArgumentParser(description="Train ranking models")
    parser.parse_args()
    cfg = get_base_config()
    processed = _require_processed_data()
    features_dir = cfg.resolve_path(cfg.paths.features_dir)
    _, items, train, val, _ = load_splits(processed)
    item_features = __import__("pandas").read_parquet(features_dir / "item_features_batch.parquet")
    from recsys.ranking.trainer import train_ranking_models

    train_ranking_models(train, val, items, item_features, cfg.resolve_path(cfg.paths.models_dir) / "ranking")


def run_simulator() -> None:
    parser = argparse.ArgumentParser(description="Run marketplace simulator")
    parser.parse_args()
    cfg = get_base_config()
    processed = _require_processed_data()
    interactions, items, _, _, _ = load_splits(processed)
    log_dir = ensure_dir(cfg.resolve_path(cfg.paths.simulator_logs_dir))
    runner = build_default_runner(items, interactions, log_dir)
    sim_cfg = __import__("recsys.config", fromlist=["load_yaml"]).load_yaml("simulator.yaml")
    runner.run(sim_cfg["simulation"]["num_days"], sim_cfg["simulation"]["sessions_per_day"])
    path = runner.logger.flush()
    print(f"Simulator logs written to {path}")


def evaluate() -> None:
    parser = argparse.ArgumentParser(description="Run offline and OPE evaluation")
    parser.parse_args()
    cfg = get_base_config()
    log_path = cfg.resolve_path(cfg.paths.simulator_logs_dir) / "simulator_logs.parquet"
    if not log_path.exists():
        run_simulator()
    import pandas as pd

    from recsys.evaluation.ope import compare_ope_estimators

    logs = pd.read_parquet(log_path)
    feature_cols = ["position", "relevance", "score"]
    results = compare_ope_estimators(logs, feature_cols)
    print(results)


def measure_skew() -> None:
    _require_processed_data()
    interactions, items = load_processed_data()
    cfg = get_base_config()
    from recsys.features.batch import build_batch_features

    batch_user, _ = build_batch_features(interactions, items)
    online = OnlineFeaturePipeline()
    online_user, _ = online.build_online_snapshot(interactions, items)
    report = measure_feature_skew(batch_user, online_user)
    out = cfg.resolve_path(cfg.paths.features_dir) / "skew_report.csv"
    ensure_dir(out.parent)
    report.to_csv(out, index=False)
    print(report)
    print(f"Saved skew report to {out}")
