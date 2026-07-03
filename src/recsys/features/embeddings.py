"""Content embedding helpers for item features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EMBEDDING_DIM = 32
EMBEDDING_COLS = [f"content_emb_{i}" for i in range(EMBEDDING_DIM)]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _text_for_item(row: pd.Series, top_review: str = "") -> str:
    title = str(row.get("title", "")).strip()
    description = str(row.get("description", "")).strip()
    parts = [p for p in [title, description, top_review[:256]] if p]
    return ". ".join(parts).strip(". ")


def _build_top_review_map(reviews: pd.DataFrame) -> dict[str, str]:
    """Return item_id → most helpful review text snippet."""
    if "review_text" not in reviews.columns:
        return {}
    sort_col = "helpful_vote" if "helpful_vote" in reviews.columns else None
    df = reviews[["item_id", "review_text"] + ([sort_col] if sort_col else [])].copy()
    df["review_text"] = df["review_text"].fillna("").astype(str)
    if sort_col:
        df = df.sort_values(sort_col, ascending=False)
    top = df.drop_duplicates(subset=["item_id"])
    return dict(zip(top["item_id"], top["review_text"]))


def compute_content_embeddings(
    items: pd.DataFrame,
    reviews: pd.DataFrame | None = None,
    output_dir: Path | None = None,
    batch_size: int = 64,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Compute truncated content embeddings for each item.

    When ``reviews`` is provided, the most helpful review text (sorted by
    ``helpful_vote`` descending) is appended to each item's title+description
    before encoding, giving the model a richer signal from real user language.
    """
    from sentence_transformers import SentenceTransformer

    top_review_map = _build_top_review_map(reviews) if reviews is not None else {}
    texts = [
        _text_for_item(row, top_review_map.get(str(row.get("item_id", "")), ""))
        for _, row in items.iterrows()
    ]
    model = SentenceTransformer(MODEL_NAME)
    full_embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    full_embeddings = np.asarray(full_embeddings, dtype=np.float32)

    if full_embeddings.shape[1] > EMBEDDING_DIM:
        truncated = full_embeddings[:, :EMBEDDING_DIM]
    else:
        pad_width = EMBEDDING_DIM - full_embeddings.shape[1]
        truncated = np.pad(full_embeddings, ((0, 0), (0, pad_width)))

    embedding_cols = [f"content_emb_{i}" for i in range(EMBEDDING_DIM)]
    embedding_df = pd.DataFrame(truncated, columns=embedding_cols)
    embedding_df.insert(0, "item_id", items["item_id"].values)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "item_embeddings.npy", full_embeddings)

    return embedding_df, full_embeddings


def mean_embedding_for_items(
    item_ids: list[str],
    items: pd.DataFrame,
    full_embeddings: np.ndarray | None = None,
    embedding_df: pd.DataFrame | None = None,
) -> np.ndarray:
    """Return mean 32-dim embedding vector for a list of item IDs."""
    if not item_ids:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    if embedding_df is not None:
        rows = embedding_df[embedding_df["item_id"].isin(item_ids)]
        if rows.empty:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)
        cols = [c for c in rows.columns if c.startswith("content_emb_")]
        return rows[cols].mean(axis=0).to_numpy(dtype=np.float32)

    if full_embeddings is not None:
        id_to_idx = {item_id: i for i, item_id in enumerate(items["item_id"].tolist())}
        vectors = [full_embeddings[id_to_idx[i]][:EMBEDDING_DIM] for i in item_ids if i in id_to_idx]
        if not vectors:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)
        return np.mean(vectors, axis=0).astype(np.float32)

    return np.zeros(EMBEDDING_DIM, dtype=np.float32)
