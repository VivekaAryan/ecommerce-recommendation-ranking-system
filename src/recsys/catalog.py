"""Catalog presentation helpers for the storefront UI."""

from __future__ import annotations

import pandas as pd

PLACEHOLDER_IMAGE = "https://placehold.co/600x600/f5f5f5/666666?text=No+Image"


def _row_item_id(row: pd.Series, item_id: str | None = None) -> str:
    if item_id is not None:
        return str(item_id)
    if "item_id" in row.index and not pd.isna(row.get("item_id")):
        return str(row["item_id"])
    return str(row.name) if row.name is not None else ""


def _display_category(row: pd.Series) -> str:
    main_category = row.get("main_category")
    if main_category is not None and not pd.isna(main_category) and str(main_category).strip():
        return str(main_category)
    return str(row.get("category", "Unknown"))


def resolve_image_url(row: pd.Series, item_id: str | None = None) -> str:
    stored = row.get("image_url")
    if stored is not None and not pd.isna(stored) and str(stored).strip():
        return str(stored)
    return PLACEHOLDER_IMAGE


def resolve_title(row: pd.Series, item_id: str | None = None) -> str:
    title = str(row.get("title", "")).strip()
    iid = _row_item_id(row, item_id)
    return title or iid


def resolve_description(row: pd.Series, item_id: str | None = None) -> str:
    desc = str(row.get("description", "")).strip()
    return desc[:200]


def item_image_url(item_id: str, category: str, stored_url: str | None = None) -> str:
    if stored_url and str(stored_url).strip():
        return str(stored_url)
    return PLACEHOLDER_IMAGE


def item_to_card(row: pd.Series, item_id: str | None = None) -> dict:
    category = _display_category(row)
    price = row.get("price")
    iid = _row_item_id(row, item_id)
    return {
        "item_id": iid,
        "title": resolve_title(row, item_id),
        "category": category,
        "price": None if pd.isna(price) else round(float(price), 2),
        "description": resolve_description(row, item_id),
        "image_url": resolve_image_url(row, item_id),
    }


def items_to_cards(items: pd.DataFrame) -> list[dict]:
    return [item_to_card(row) for _, row in items.iterrows()]
