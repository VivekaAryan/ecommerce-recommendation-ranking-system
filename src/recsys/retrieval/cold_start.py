"""Cold-start retrieval fallbacks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from recsys.retrieval.ann_index import ANNIndex
from recsys.retrieval.two_tower import TwoTowerModel


class ColdStartRetriever:
    def __init__(
        self,
        model: TwoTowerModel,
        ann_index: ANNIndex,
        items_df: pd.DataFrame,
        item_features: np.ndarray,
        item_id_to_idx: dict[str, int],
    ) -> None:
        self.model = model
        self.ann_index = ann_index
        self.items_df = items_df
        self.item_features = item_features
        self.item_id_to_idx = item_id_to_idx

    def retrieve_for_user(
        self,
        history_item_ids: list[str] | None,
        history_tensor: torch.Tensor | None,
        top_k: int = 100,
        preferred_category: str | None = None,
    ) -> list[str]:
        if history_tensor is not None and history_tensor.numel() > 0:
            self.model.eval()
            with torch.no_grad():
                user_emb = self.model.encode_users(history_tensor).cpu().numpy()
            results, _ = self.ann_index.search(user_emb, top_k=top_k)
            return results[0]

        return self._content_fallback(top_k, preferred_category)

    def retrieve_new_item_neighbors(self, item_id: str, top_k: int = 20) -> list[str]:
        idx = self.item_id_to_idx.get(item_id)
        if idx is None:
            return []
        item_ids_tensor = torch.tensor([idx])
        item_feat = torch.tensor(self.item_features[idx : idx + 1], dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            item_emb = self.model.encode_items(item_ids_tensor, item_feat).cpu().numpy()
        results, _ = self.ann_index.search(item_emb, top_k=top_k + 1)
        return [i for i in results[0] if i != item_id][:top_k]

    def _content_fallback(self, top_k: int, preferred_category: str | None) -> list[str]:
        df = self.items_df.copy()
        if preferred_category:
            cat_df = df[df["category"] == preferred_category]
            if not cat_df.empty:
                df = cat_df
        ranked = df.sort_values("price").head(top_k)
        return ranked["item_id"].tolist()
