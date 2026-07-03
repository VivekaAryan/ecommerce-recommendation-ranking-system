"""Small in-memory catalog fixtures for unit tests (not production data)."""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = [
    "Appliances",
    "Books",
    "Electronics",
]


def make_test_items(num_items: int = 20, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(num_items):
        category = CATEGORIES[i % len(CATEGORIES)]
        rows.append(
            {
                "item_id": f"B{i:08d}",
                "title": f"Test Product {i}",
                "description": f"Description for test product {i}",
                "category": category,
                "main_category": category,
                "price": round(float(rng.uniform(19.99, 499.99)), 2),
                "image_url": f"https://m.media-amazon.com/images/I/test{i:04d}.jpg",
                "average_rating": round(float(rng.uniform(3.5, 5.0)), 2),
                "rating_number": int(rng.integers(10, 500)),
                "also_buy": [],
                "also_view": [],
                "bought_together": [],
            }
        )
    return pd.DataFrame(rows)


def make_test_interactions(
    items: pd.DataFrame,
    num_interactions: int = 100,
    num_users: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    users = [f"amazon_user_{i}" for i in range(num_users)]
    item_ids = items["item_id"].tolist()
    main_categories = items["main_category"].tolist()
    timestamps = rng.integers(1_500_000_000, 1_600_000_000, size=num_interactions)
    verified = rng.random(size=num_interactions) > 0.3
    df = pd.DataFrame(
        {
            "user_id": rng.choice(users, size=num_interactions),
            "item_id": rng.choice(item_ids, size=num_interactions),
            "rating": rng.integers(1, 6, size=num_interactions),
            "timestamp": timestamps,
            "verified_purchase": verified,
            "review_text": [f"Review text {i}" for i in range(num_interactions)],
            "helpful_vote": rng.integers(0, 20, size=num_interactions),
            "main_category": rng.choice(main_categories, size=num_interactions),
        }
    )
    return df.drop_duplicates(subset=["user_id", "item_id", "timestamp"])
