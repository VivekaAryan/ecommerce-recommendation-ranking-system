"""Business logic for the testing dashboard API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from recsys.catalog import item_to_card, items_to_cards
from recsys.config import get_base_config, load_yaml
from recsys.data.download import prepare_dataset
from recsys.data.splits import load_splits
from recsys.evaluation.interference import cluster_randomized_test, naive_ab_test, switchback_test
from recsys.evaluation.ope import compare_ope_estimators
from recsys.features.batch import load_processed_data, save_batch_features
from recsys.features.online import OnlineFeaturePipeline, measure_feature_skew
from recsys.simulator.runner import build_default_runner
from recsys.utils import ensure_dir


def _invalidate_recommender() -> None:
    from recsys.serving.recommender import get_recommendation_service

    get_recommendation_service().invalidate()


def _artifact(path: Path, name: str, detail: str = "") -> dict[str, Any]:
    return {
        "name": name,
        "exists": path.exists(),
        "path": str(path),
        "detail": detail if path.exists() else "Not found",
    }


def get_system_status() -> dict[str, Any]:
    cfg = get_base_config()
    processed = cfg.resolve_path(cfg.paths.processed_dir)
    features = cfg.resolve_path(cfg.paths.features_dir)
    models = cfg.resolve_path(cfg.paths.models_dir)
    logs = cfg.resolve_path(cfg.paths.simulator_logs_dir)

    artifacts = [
        _artifact(processed / "interactions.parquet", "interactions"),
        _artifact(processed / "items.parquet", "items"),
        _artifact(features / "user_features_batch.parquet", "user_features"),
        _artifact(features / "item_features_batch.parquet", "item_features"),
        _artifact(models / "retrieval" / "two_tower.pt", "retrieval_model"),
        _artifact(models / "retrieval" / "item_index.faiss", "ann_index"),
        _artifact(models / "ranking" / "lgbm_ranker.txt", "lgbm_ranker"),
        _artifact(logs / "simulator_logs.parquet", "simulator_logs"),
        _artifact(features / "skew_report.csv", "skew_report"),
    ]

    dataset = None
    stats_path = processed / "stats.json"
    if stats_path.exists():
        dataset = json.loads(stats_path.read_text())

    ready = all(a["exists"] for a in artifacts[:2])
    return {"ready": ready, "artifacts": artifacts, "dataset": dataset}


def list_users_from_dataset(limit: int = 50) -> list[dict[str, Any]]:
    """List users from interactions parquet without loading ML models."""
    cfg = get_base_config()
    path = cfg.resolve_path(cfg.paths.processed_dir) / "interactions.parquet"
    if not path.exists():
        raise FileNotFoundError("Dataset not prepared. Run the data pipeline first.")
    interactions = pd.read_parquet(path)
    counts = interactions["user_id"].value_counts().head(limit)
    return [
        {"user_id": str(uid), "interaction_count": int(count)} for uid, count in counts.items()
    ]


def _load_items_df() -> pd.DataFrame:
    cfg = get_base_config()
    path = cfg.resolve_path(cfg.paths.processed_dir) / "items.parquet"
    if not path.exists():
        raise FileNotFoundError("Dataset not prepared. Run the data pipeline first.")
    items = pd.read_parquet(path)
    if "image_url" in items.columns:
        items = items[items["image_url"].notna() & (items["image_url"].astype(str).str.strip() != "")]
    return items


def _category_column(items: pd.DataFrame) -> str:
    return "main_category" if "main_category" in items.columns else "category"


def get_catalog(
    limit: int = 48,
    offset: int = 0,
    category: str | None = None,
) -> dict[str, Any]:
    items = _load_items_df()
    category_col = _category_column(items)
    if category:
        items = items[items[category_col] == category]
    total = len(items)
    page = items.iloc[offset : offset + limit]
    all_items = _load_items_df()
    categories = sorted(all_items[category_col].dropna().unique().tolist())
    return {
        "products": items_to_cards(page),
        "categories": categories,
        "total": total,
    }


def get_user_history_products(user_id: str, limit: int = 12) -> dict[str, Any]:
    cfg = get_base_config()
    interactions_path = cfg.resolve_path(cfg.paths.processed_dir) / "interactions.parquet"
    if not interactions_path.exists():
        raise FileNotFoundError("Dataset not prepared. Run the data pipeline first.")
    interactions = pd.read_parquet(interactions_path)
    user_rows = interactions[interactions["user_id"] == user_id].sort_values("timestamp")
    item_ids = user_rows["item_id"].drop_duplicates().tolist()[-limit:]
    items = _load_items_df()
    products = []
    for item_id in reversed(item_ids):
        row = items[items["item_id"] == item_id]
        if not row.empty:
            products.append(item_to_card(row.iloc[0]))
    return {"user_id": user_id, "products": products}


def run_download() -> dict[str, Any]:
    paths = prepare_dataset()
    _invalidate_recommender()
    return {
        "message": "Dataset prepared",
        "paths": {k: str(v) for k, v in paths.items()},
    }


def run_build_features() -> dict[str, Any]:
    interactions, items = load_processed_data()
    cfg = get_base_config()
    paths = save_batch_features(interactions, items, cfg.resolve_path(cfg.paths.features_dir))
    _invalidate_recommender()
    return {"message": "Features built", "paths": {k: str(v) for k, v in paths.items()}}


def run_train_retrieval() -> dict[str, Any]:
    from recsys.retrieval.trainer import train_retrieval_model

    cfg = get_base_config()
    processed = cfg.resolve_path(cfg.paths.processed_dir)
    _, items, train, val, _ = load_splits(processed)
    output = cfg.resolve_path(cfg.paths.models_dir) / "retrieval"
    train_retrieval_model(train, val, items, output)
    _invalidate_recommender()
    return {"message": "Retrieval model trained", "output": str(output)}


def run_train_ranker() -> dict[str, Any]:
    from recsys.ranking.trainer import train_ranking_models

    cfg = get_base_config()
    processed = cfg.resolve_path(cfg.paths.processed_dir)
    features_dir = cfg.resolve_path(cfg.paths.features_dir)
    _, items, train, val, _ = load_splits(processed)
    item_features = pd.read_parquet(features_dir / "item_features_batch.parquet")
    output = cfg.resolve_path(cfg.paths.models_dir) / "ranking"
    train_ranking_models(train, val, items, item_features, output)
    _invalidate_recommender()
    return {"message": "Ranking models trained", "output": str(output)}


def run_simulator(
    num_days: int | None = None,
    sessions_per_day: int | None = None,
) -> dict[str, Any]:
    cfg = get_base_config()
    sim_cfg = load_yaml("simulator.yaml")
    processed = cfg.resolve_path(cfg.paths.processed_dir)
    interactions, items, _, _, _ = load_splits(processed)
    log_dir = ensure_dir(cfg.resolve_path(cfg.paths.simulator_logs_dir))

    candidate_fn = None
    retrieval_dir = cfg.resolve_path(cfg.paths.models_dir) / "retrieval"
    if (retrieval_dir / "two_tower.pt").exists():
        from recsys.retrieval.simulator_integration import load_retrieval_for_simulator

        candidate_fn = load_retrieval_for_simulator(items, interactions, retrieval_dir)

    runner = build_default_runner(items, interactions, log_dir, candidate_fn=candidate_fn)
    days = num_days or sim_cfg["simulation"]["num_days"]
    sessions = sessions_per_day or sim_cfg["simulation"]["sessions_per_day"]
    logs = runner.run(days, sessions)
    path = runner.logger.flush()
    return {
        "message": f"Simulator completed: {len(logs)} impressions",
        "log_path": str(path),
        "impressions": len(logs),
        "click_rate": float(logs["clicked"].mean()) if len(logs) else 0.0,
    }


def run_evaluate() -> dict[str, Any]:
    cfg = get_base_config()
    log_path = cfg.resolve_path(cfg.paths.simulator_logs_dir) / "simulator_logs.parquet"
    if not log_path.exists():
        run_simulator()
        log_path = cfg.resolve_path(cfg.paths.simulator_logs_dir) / "simulator_logs.parquet"

    logs = pd.read_parquet(log_path)
    feature_cols = ["position", "relevance", "score"]
    ope = compare_ope_estimators(logs, feature_cols)

    logs = logs.copy()
    logs["policy_id_treatment"] = logs["policy_id"] + "_treatment"
    treated = logs.copy()
    treated["policy_id"] = treated["policy_id"] + "_treatment"
    ab_logs = pd.concat([logs, treated], ignore_index=True)
    naive = naive_ab_test(ab_logs)
    switchback = switchback_test(ab_logs)
    cluster = cluster_randomized_test(ab_logs, cluster_col="user_id")

    return {
        "message": "Evaluation complete",
        "ope": ope,
        "interference": {
            "naive_ab": naive.treatment_effect,
            "switchback": switchback.treatment_effect,
            "cluster_randomized": cluster.treatment_effect,
        },
        "simulator_summary": {
            "impressions": len(logs),
            "click_rate": float(logs["clicked"].mean()),
            "purchase_rate": float(logs["purchased"].mean()),
            "unique_items": int(logs["item_id"].nunique()),
        },
    }


def run_measure_skew() -> dict[str, Any]:
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
    return {
        "message": "Skew report generated",
        "report": report.to_dict(orient="records"),
        "path": str(out),
    }


def get_metrics() -> dict[str, Any]:
    cfg = get_base_config()
    result: dict[str, Any] = {}

    skew_path = cfg.resolve_path(cfg.paths.features_dir) / "skew_report.csv"
    if skew_path.exists():
        result["skew"] = pd.read_csv(skew_path).to_dict(orient="records")

    log_path = cfg.resolve_path(cfg.paths.simulator_logs_dir) / "simulator_logs.parquet"
    if log_path.exists():
        logs = pd.read_parquet(log_path)
        result["ope"] = compare_ope_estimators(logs, ["position", "relevance", "score"])
        result["simulator_summary"] = {
            "impressions": len(logs),
            "click_rate": round(float(logs["clicked"].mean()), 4),
            "purchase_rate": round(float(logs["purchased"].mean()), 4),
            "unique_items": int(logs["item_id"].nunique()),
            "unique_users": int(logs["user_id"].nunique()),
        }

    return result


def get_simulator_logs(limit: int = 100, offset: int = 0) -> dict[str, Any]:
    cfg = get_base_config()
    log_path = cfg.resolve_path(cfg.paths.simulator_logs_dir) / "simulator_logs.parquet"
    if not log_path.exists():
        return {"total": 0, "rows": []}
    logs = pd.read_parquet(log_path)
    page = logs.iloc[offset : offset + limit]
    return {"total": len(logs), "rows": page.to_dict(orient="records")}


TASK_HANDLERS = {
    "download": lambda p: run_download(),
    "features": lambda p: run_build_features(),
    "train_retrieval": lambda p: run_train_retrieval(),
    "train_ranker": lambda p: run_train_ranker(),
    "simulator": lambda p: run_simulator(p.num_days, p.sessions_per_day),
    "evaluate": lambda p: run_evaluate(),
    "measure_skew": lambda p: run_measure_skew(),
}
