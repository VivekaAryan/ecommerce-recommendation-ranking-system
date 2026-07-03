"""Retrieval training loop and dataset helpers."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from recsys.config import get_base_config, load_yaml
from recsys.experiments.tracker import ExperimentTracker
from recsys.features.item_features import build_item_feature_matrix
from recsys.retrieval.ann_index import ANNIndex, benchmark_ann
from recsys.retrieval.embedding_versioning import (
    apply_alignment,
    procrustes_align,
    select_anchor_indices,
)
from recsys.retrieval.two_tower import (
    TwoTowerConfig,
    TwoTowerModel,
    compute_inbatch_logits,
    logq_corrected_loss,
)
from recsys.utils import ensure_dir, get_device, set_seed

VERIFIED_SAMPLE_WEIGHT = 2.0


class RetrievalDataset(Dataset):
    def __init__(
        self,
        interactions: pd.DataFrame,
        item_id_to_idx: dict[str, int],
        item_features: np.ndarray,
        history_length: int = 50,
    ) -> None:
        self.history_length = history_length
        self.item_id_to_idx = item_id_to_idx
        self.item_features = item_features
        self.samples: list[tuple[list[int], int, float]] = []
        history: dict[str, list[str]] = defaultdict(list)
        for row in interactions.sort_values("timestamp").itertuples():
            item_idx = item_id_to_idx.get(row.item_id)
            if item_idx is None:
                continue
            hist = [item_id_to_idx[i] for i in history[row.user_id][-history_length:] if i in item_id_to_idx]
            if hist:
                weight = VERIFIED_SAMPLE_WEIGHT if getattr(row, "verified_purchase", False) else 1.0
                self.samples.append((hist, item_idx, weight))
            history[row.user_id].append(row.item_id)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hist, pos_item, weight = self.samples[idx]
        hist_tensor = torch.zeros(self.history_length, dtype=torch.long)
        hist_tensor[-len(hist) :] = torch.tensor(hist, dtype=torch.long)
        return (
            hist_tensor.unsqueeze(0),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(self.item_features[pos_item], dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32),
        )


def _build_mappings(
    items: pd.DataFrame,
    item_features: pd.DataFrame | None = None,
) -> tuple[dict[str, int], np.ndarray]:
    item_ids = items["item_id"].tolist()
    item_id_to_idx = {item_id: i + 1 for i, item_id in enumerate(item_ids)}
    feature_matrix = build_item_feature_matrix(items, item_features)
    return item_id_to_idx, feature_matrix


def train_retrieval_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    items_df: pd.DataFrame,
    output_dir: Path,
    epochs: int | None = None,
    item_features: pd.DataFrame | None = None,
) -> tuple[TwoTowerModel, ANNIndex]:
    cfg = load_yaml("retrieval.yaml")
    base_cfg = get_base_config()
    set_seed(base_cfg.seed)
    device = get_device()
    output_dir = ensure_dir(output_dir)

    item_id_to_idx, item_features_np = _build_mappings(items_df, item_features)
    popularity = train_df["item_id"].value_counts()
    pop_tensor = torch.tensor(
        [popularity.get(item_id, 1) for item_id in items_df["item_id"]],
        dtype=torch.float32,
        device=device,
    )

    model_cfg = TwoTowerConfig(
        num_users=train_df["user_id"].nunique() + 1,
        num_items=len(item_id_to_idx) + 1,
        embedding_dim=cfg["model"]["embedding_dim"],
        user_hidden_dim=cfg["model"]["user_hidden_dim"],
        item_hidden_dim=cfg["model"]["item_hidden_dim"],
        history_length=cfg["model"]["history_length"],
        item_feature_dim=item_features_np.shape[1],
        dropout=cfg["model"]["dropout"],
        temperature=cfg["training"]["temperature"],
    )
    model = TwoTowerModel(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    train_ds = RetrievalDataset(train_df, item_id_to_idx, item_features_np, cfg["model"]["history_length"])
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)

    tracker = ExperimentTracker()
    num_epochs = epochs or cfg["training"]["epochs"]
    with tracker.run("train_retrieval"):
        tracker.log_params({"epochs": num_epochs, "batch_size": cfg["training"]["batch_size"]})
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            for histories, pos_items, pos_feats, sample_weights in train_loader:
                histories = histories.squeeze(1).to(device)
                pos_items = pos_items.to(device)
                pos_feats = pos_feats.to(device)
                sample_weights = sample_weights.to(device)
                user_emb = model.encode_users(histories)
                item_emb = model.encode_items(pos_items, pos_feats)
                logits = compute_inbatch_logits(user_emb, item_emb)
                batch_pop = pop_tensor[pos_items - 1]
                loss = logq_corrected_loss(
                    logits,
                    batch_pop,
                    temperature=cfg["training"]["temperature"],
                    sample_weights=sample_weights,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            tracker.log_metrics({"train_loss": total_loss / max(len(train_loader), 1)}, step=epoch)

    model.eval()
    all_item_ids = items_df["item_id"].tolist()
    all_indices = torch.tensor([item_id_to_idx[i] for i in all_item_ids], device=device)
    all_feats = torch.tensor(item_features_np[all_indices.cpu().numpy() - 1], dtype=torch.float32, device=device)
    with torch.no_grad():
        item_embeddings = model.encode_items(all_indices, all_feats).cpu().numpy()

    ann = ANNIndex(cfg["model"]["embedding_dim"], index_type=cfg["ann"]["index_type"])
    ann.build(item_embeddings, all_item_ids)
    ann.save(output_dir / "item_index.faiss")

    val_samples = RetrievalDataset(val_df, item_id_to_idx, item_features_np, cfg["model"]["history_length"])
    if len(val_samples) > 0:
        histories, _, _, _ = val_samples[0]
        query = model.encode_users(histories.to(device)).detach().cpu().numpy()
        benchmark = benchmark_ann(item_embeddings, all_item_ids, query, [[all_item_ids[0]]], k=50)
        tracker.log_metrics({f"ann_{r.index_type}_recall": r.recall_at_k for r in benchmark})

    anchor_idx = select_anchor_indices(len(all_item_ids), cfg["embedding_versioning"]["num_anchors"])
    np.save(output_dir / "anchor_indices.npy", anchor_idx)
    old_anchors = item_embeddings[anchor_idx]
    new_anchors = item_embeddings[anchor_idx]
    transform = procrustes_align(old_anchors, new_anchors)
    np.save(output_dir / "embedding_alignment.npy", transform)
    aligned = apply_alignment(item_embeddings, transform)
    np.save(output_dir / "item_embeddings.npy", aligned)

    torch.save(model.state_dict(), output_dir / "two_tower.pt")
    return model, ann
