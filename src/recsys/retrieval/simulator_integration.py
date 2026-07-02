"""Wire trained retrieval model into simulator candidate generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from recsys.config import get_base_config, load_yaml
from recsys.retrieval.ann_index import ANNIndex
from recsys.retrieval.two_tower import TwoTowerConfig, TwoTowerModel
from recsys.simulator.environment import MarketplaceEnvironment


def build_retrieval_candidate_fn(
    model: TwoTowerModel,
    ann_index: ANNIndex,
    items_df: pd.DataFrame,
    item_features: np.ndarray,
    item_id_to_idx: dict[str, int],
    interactions_df: pd.DataFrame,
    top_k: int = 500,
):
    """Return a candidate function compatible with SimulatorRunner."""
    history_length = load_yaml("retrieval.yaml")["model"]["history_length"]
    user_history: dict[str, list[str]] = {}
    for row in interactions_df.sort_values("timestamp").itertuples():
        user_history.setdefault(row.user_id, []).append(row.item_id)

    device = next(model.parameters()).device

    def candidate_fn(env: MarketplaceEnvironment, user_id: str) -> list[str]:
        hist_ids = user_history.get(user_id, [])[-history_length:]
        if hist_ids:
            hist_idx = [item_id_to_idx[i] for i in hist_ids if i in item_id_to_idx]
            if hist_idx:
                hist_tensor = torch.zeros(1, history_length, dtype=torch.long, device=device)
                hist_tensor[0, -len(hist_idx) :] = torch.tensor(hist_idx, device=device)
                model.eval()
                with torch.no_grad():
                    user_emb = model.encode_users(hist_tensor).cpu().numpy()
                results, _ = ann_index.search(user_emb, top_k=top_k)
                return results[0]
        return list(env.items.keys())[:top_k]

    return candidate_fn


def load_retrieval_for_simulator(
    items_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    model_dir: Path | None = None,
):
    """Load trained retrieval artifacts and return a candidate function."""
    cfg = get_base_config()
    model_dir = model_dir or cfg.resolve_path(cfg.paths.models_dir) / "retrieval"
    retrieval_cfg = load_yaml("retrieval.yaml")

    item_ids = items_df["item_id"].tolist()
    item_id_to_idx = {item_id: i + 1 for i, item_id in enumerate(item_ids)}
    item_features = np.zeros((len(item_ids), 16), dtype=np.float32)

    model_cfg = TwoTowerConfig(
        num_users=interactions_df["user_id"].nunique() + 1,
        num_items=len(item_id_to_idx) + 1,
        embedding_dim=retrieval_cfg["model"]["embedding_dim"],
        user_hidden_dim=retrieval_cfg["model"]["user_hidden_dim"],
        item_hidden_dim=retrieval_cfg["model"]["item_hidden_dim"],
        history_length=retrieval_cfg["model"]["history_length"],
        item_feature_dim=16,
    )
    model = TwoTowerModel(model_cfg)
    state_path = model_dir / "two_tower.pt"
    if state_path.exists():
        model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))

    ann = ANNIndex(retrieval_cfg["model"]["embedding_dim"])
    index_path = model_dir / "item_index.faiss"
    if index_path.exists():
        ann.load(index_path)
    else:
        rng = np.random.default_rng(cfg.seed)
        emb = rng.normal(size=(len(item_ids), retrieval_cfg["model"]["embedding_dim"])).astype(
            np.float32
        )
        ann.build(emb, item_ids)

    return build_retrieval_candidate_fn(
        model, ann, items_df, item_features, item_id_to_idx, interactions_df
    )
