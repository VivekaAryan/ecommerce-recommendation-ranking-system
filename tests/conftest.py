"""Pytest hooks and shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from recsys.config import get_base_config
from recsys.data.item_links import build_item_links, save_item_links
from recsys.data.splits import assign_time_split, save_splits
from recsys.env import configure_runtime_env
from tests.fixtures.catalog import make_test_interactions, make_test_items

configure_runtime_env()


def write_test_processed_data(
    processed_dir: Path,
    num_interactions: int = 500,
    num_items: int = 50,
    num_users: int = 20,
    seed: int = 42,
) -> dict[str, Path]:
    items = make_test_items(num_items=num_items, seed=seed)
    reviews = make_test_interactions(items, num_interactions=num_interactions, num_users=num_users, seed=seed)
    cfg = get_base_config()
    reviews = assign_time_split(reviews, cfg.splits.train_end, cfg.splits.val_end)
    paths = save_splits(reviews, items, processed_dir)
    links = build_item_links(items)
    link_path = save_item_links(links, processed_dir / "item_links.parquet")
    paths["item_links"] = link_path
    return paths


@pytest.fixture
def processed_test_data(tmp_path, monkeypatch):
    cfg = get_base_config()
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_path(relative: str) -> Path:
        if relative == cfg.paths.processed_dir:
            return processed_dir
        return tmp_path / relative

    monkeypatch.setattr(
        "recsys.config.get_base_config",
        lambda: type(
            "Cfg",
            (),
            {
                **cfg.model_dump(),
                "resolve_path": _resolve_path,
            },
        )(),
    )
    paths = write_test_processed_data(processed_dir)
    return paths
