"""Ranking model training orchestration."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from recsys.config import get_base_config, load_yaml
from recsys.experiments.tracker import ExperimentTracker
from recsys.ranking.lgbm_ranker import LightGBMRanker, build_ranking_features
from recsys.ranking.multitask import MultiTaskRanker, MultiTaskWeights
from recsys.ranking.sequential import SequentialRanker
from recsys.utils import ensure_dir, get_device, set_seed


class RankingDataset(Dataset):
    def __init__(self, interactions: pd.DataFrame, item_id_to_idx: dict[str, int], seq_len: int = 50) -> None:
        self.seq_len = seq_len
        self.item_id_to_idx = item_id_to_idx
        self.samples: list[tuple[list[int], int, float]] = []
        history: dict[str, list[str]] = defaultdict(list)
        for row in interactions.sort_values("timestamp").itertuples():
            if row.item_id not in item_id_to_idx:
                continue
            hist = [item_id_to_idx[i] for i in history[row.user_id][-seq_len:] if i in item_id_to_idx]
            if hist:
                label_click = 1.0
                label_value = 0.0 if pd.isna(getattr(row, "price", np.nan)) else float(getattr(row, "price", 0.0))
                self.samples.append((hist, item_id_to_idx[row.item_id], label_click, label_value))
            history[row.user_id].append(row.item_id)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        hist, pos_item, click, value = self.samples[idx]
        seq = torch.zeros(self.seq_len, dtype=torch.long)
        seq[-len(hist) :] = torch.tensor(hist, dtype=torch.long)
        return seq, torch.tensor(pos_item), torch.tensor(click), torch.tensor(value)


def train_ranking_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    items_df: pd.DataFrame,
    item_features: pd.DataFrame,
    output_dir: Path,
) -> tuple[SequentialRanker, MultiTaskRanker, LightGBMRanker]:
    cfg = load_yaml("ranking.yaml")
    base_cfg = get_base_config()
    set_seed(base_cfg.seed)
    device = get_device()
    output_dir = ensure_dir(output_dir)

    item_ids = items_df["item_id"].tolist()
    item_id_to_idx = {item_id: i + 1 for i, item_id in enumerate(item_ids)}
    num_items = len(item_id_to_idx) + 1

    seq_model = SequentialRanker(
        num_items=num_items,
        hidden_dim=cfg["sequential"]["hidden_dim"],
        num_layers=cfg["sequential"]["num_layers"],
        num_heads=cfg["sequential"]["num_heads"],
        max_seq_length=cfg["sequential"]["max_seq_length"],
        dropout=cfg["sequential"]["dropout"],
    ).to(device)

    multitask = MultiTaskRanker(
        input_dim=cfg["sequential"]["hidden_dim"] + 6,
        hidden_dim=cfg["sequential"]["hidden_dim"],
    ).to(device)

    train_ds = RankingDataset(train_df, item_id_to_idx, cfg["sequential"]["max_seq_length"])
    loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
    optimizer = torch.optim.Adam(
        list(seq_model.parameters()) + list(multitask.parameters()),
        lr=cfg["training"]["learning_rate"],
    )
    MultiTaskWeights(**cfg["multitask"]["head_weights"])

    tracker = ExperimentTracker()
    with tracker.run("train_ranking"):
        for epoch in range(cfg["training"]["epochs"]):
            seq_model.train()
            multitask.train()
            total_loss = 0.0
            for seq, pos_item, click, value in loader:
                seq = seq.to(device)
                pos_item = pos_item.to(device)
                click = click.to(device)
                value = value.to(device)
                user_repr = seq_model(seq)
                candidate_emb = seq_model.item_embedding(pos_item)
                deep_feat = torch.cat([user_repr, candidate_emb], dim=-1)
                side = torch.zeros(deep_feat.size(0), 6, device=device)
                outputs = multitask(torch.cat([deep_feat, side], dim=-1))
                loss = (
                    F.binary_cross_entropy(outputs["click"], click)
                    + F.binary_cross_entropy(outputs["purchase"], click * 0.5)
                    + F.mse_loss(outputs["expected_value"], value)
                    + F.binary_cross_entropy(outputs["engagement"], click * 0.3)
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            tracker.log_metrics({"train_loss": total_loss / max(len(loader), 1)}, step=epoch)

    lgbm = LightGBMRanker(cfg["lgbm"])
    features = []
    labels = []
    groups = []
    for _user_id, user_df in val_df.groupby("user_id"):
        candidates = user_df["item_id"].tolist()[: cfg["prerank"]["top_k"]]
        retrieval_scores = {c: 1.0 for c in candidates}
        deep_scores = {c: 0.5 for c in candidates}
        feat, valid_ids = build_ranking_features(candidates, retrieval_scores, deep_scores, item_features)
        if len(valid_ids) == 0:
            continue
        features.append(feat)
        labels.append(np.ones(len(valid_ids)))
        groups.append(len(valid_ids))
    if features:
        lgbm.fit(np.vstack(features), np.concatenate(labels), groups)

    torch.save(seq_model.state_dict(), output_dir / "sequential_ranker.pt")
    torch.save(multitask.state_dict(), output_dir / "multitask_ranker.pt")
    lgbm.save(output_dir / "lgbm_ranker.txt")
    return seq_model, multitask, lgbm
