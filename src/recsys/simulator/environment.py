"""Marketplace environment with inventory and interference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CatalogItem:
    item_id: str
    price: float
    category: str
    inventory: int
    attention_score: float = 1.0


@dataclass
class SyntheticUser:
    user_id: str
    preference_vector: np.ndarray
    category_affinity: str


@dataclass
class MarketplaceEnvironment:
    items: dict[str, CatalogItem]
    users: dict[str, SyntheticUser]
    item_embeddings: dict[str, np.ndarray]
    attention_decay_rate: float = 0.01

    @classmethod
    def from_dataframes(
        cls,
        items_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        num_synthetic_users: int = 200,
        default_inventory: int = 100,
        attention_decay_rate: float = 0.01,
        seed: int = 42,
    ) -> MarketplaceEnvironment:
        rng = np.random.default_rng(seed)
        items: dict[str, CatalogItem] = {}
        embeddings: dict[str, np.ndarray] = {}

        for row in items_df.itertuples():
            items[row.item_id] = CatalogItem(
                item_id=row.item_id,
                price=float(row.price) if not pd.isna(row.price) else 50.0,
                category=str(row.category),
                inventory=default_inventory,
            )
            embeddings[row.item_id] = rng.normal(size=32)
            embeddings[row.item_id] /= np.linalg.norm(embeddings[row.item_id]) + 1e-8

        real_users = interactions_df["user_id"].unique().tolist()
        users: dict[str, SyntheticUser] = {}
        categories = items_df["category"].unique().tolist()

        for user_id in real_users[: max(1, len(real_users) - num_synthetic_users)]:
            hist = interactions_df[interactions_df["user_id"] == user_id]
            cat = hist.merge(items_df, on="item_id")["category"].mode()
            affinity = cat.iloc[0] if len(cat) else rng.choice(categories)
            users[user_id] = SyntheticUser(
                user_id=user_id,
                preference_vector=rng.normal(size=32),
                category_affinity=affinity,
            )

        for i in range(num_synthetic_users):
            user_id = f"synthetic_user_{i}"
            users[user_id] = SyntheticUser(
                user_id=user_id,
                preference_vector=rng.normal(size=32),
                category_affinity=rng.choice(categories),
            )

        return cls(
            items=items,
            users=users,
            item_embeddings=embeddings,
            attention_decay_rate=attention_decay_rate,
        )

    def relevance(self, user_id: str, item_id: str) -> float:
        user = self.users[user_id]
        item = self.items[item_id]
        emb = self.item_embeddings[item_id]
        base = float(np.dot(user.preference_vector, emb))
        category_bonus = 0.2 if item.category == user.category_affinity else 0.0
        inventory_penalty = 0.0 if item.inventory > 0 else -1.0
        attention = item.attention_score
        score = base + category_bonus + inventory_penalty
        return max(0.0, score * attention)

    def serve_item(self, item_id: str) -> None:
        item = self.items[item_id]
        if item.inventory > 0:
            item.inventory -= 1
        item.attention_score = max(0.1, item.attention_score - self.attention_decay_rate)

    def restock(self, amount: int | None = None) -> None:
        for item in self.items.values():
            item.inventory = amount if amount is not None else item.inventory
            item.attention_score = min(1.0, item.attention_score + 0.05)

    def catalog_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "item_id": item.item_id,
                    "price": item.price,
                    "category": item.category,
                    "inventory": item.inventory,
                    "attention_score": item.attention_score,
                }
                for item in self.items.values()
            ]
        )
