from pathlib import Path
from unittest.mock import patch

import pandas as pd

from recsys.config import CategoryConfig
from recsys.data.download import (
    _filter_image_backed_items,
    _iter_jsonl,
    _parse_meta_row,
    _parse_review_row,
    _resolve_jsonl,
    download_category_from_huggingface,
)
from recsys.data.item_links import build_item_links, neighbors_for_item


def test_iter_jsonl_reads_lines(tmp_path):
    jsonl_path = tmp_path / "sample.jsonl"
    jsonl_path.write_text(
        '{"user_id":"u1","parent_asin":"B001","rating":5,"timestamp":1}\n\n'
        '{"user_id":"u2","parent_asin":"B002","rating":4,"timestamp":2}\n',
        encoding="utf-8",
    )
    rows = list(_iter_jsonl(jsonl_path))
    assert len(rows) == 2
    assert rows[0]["parent_asin"] == "B001"


def test_resolve_jsonl_uses_local_cache(tmp_path):
    cache_dir = tmp_path / "Books"
    hf_filename = "raw/review_categories/Books.jsonl"
    local_path = cache_dir / hf_filename
    local_path.parent.mkdir(parents=True)
    local_path.write_text('{"user_id":"u1"}\n', encoding="utf-8")

    resolved = _resolve_jsonl(hf_filename, cache_dir)
    assert resolved == local_path


@patch("huggingface_hub.hf_hub_download")
def test_resolve_jsonl_downloads_when_missing(mock_download, tmp_path):
    cache_dir = tmp_path / "Books"
    hf_filename = "raw/review_categories/Books.jsonl"
    local_path = cache_dir / hf_filename
    local_path.parent.mkdir(parents=True)

    def _write_file(*, local_dir, filename, **kwargs):
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"user_id":"u1"}\n', encoding="utf-8")

    mock_download.side_effect = _write_file
    resolved = _resolve_jsonl(hf_filename, cache_dir)
    assert resolved == local_path
    mock_download.assert_called_once()


def test_parse_review_row_includes_extra_fields():
    row = {
        "user_id": "u1",
        "parent_asin": "B001",
        "rating": 5,
        "timestamp": 1_500_000_000,
        "verified_purchase": True,
        "text": "Great product " * 100,
        "helpful_vote": 12,
    }
    parsed = _parse_review_row(row, "Electronics")
    assert parsed["review_text"].startswith("Great product")
    assert len(parsed["review_text"]) <= 512
    assert parsed["helpful_vote"] == 12
    assert parsed["main_category"] == "Electronics"


def test_parse_meta_row_includes_links_and_ratings():
    row = {
        "parent_asin": "B001",
        "title": "Test Product",
        "description": ["A", "nice", "item"],
        "categories": ["Electronics", "Headphones"],
        "main_category": "Electronics",
        "price": "19.99",
        "average_rating": 4.5,
        "rating_number": 120,
        "also_buy": ["B002", "B003"],
        "also_view": ["B004"],
        "bought_together": ["B005"],
        "images": [{"large": "https://m.media-amazon.com/images/I/test.jpg"}],
    }
    parsed = _parse_meta_row(row, "Electronics")
    assert parsed["image_url"].startswith("https://m.media-amazon.com")
    assert parsed["also_buy"] == ["B002", "B003"]
    assert parsed["average_rating"] == 4.5


def test_filter_image_backed_items():
    items = pd.DataFrame(
        {
            "item_id": ["a", "b"],
            "image_url": ["https://m.media-amazon.com/images/I/a.jpg", ""],
        }
    )
    filtered = _filter_image_backed_items(items)
    assert len(filtered) == 1
    assert filtered.iloc[0]["item_id"] == "a"


def test_build_item_links_normalizes_edges():
    items = pd.DataFrame(
        {
            "item_id": ["A", "B", "C"],
            "also_buy": [["B"], [], ["A"]],
            "also_view": [[], ["A"], []],
            "bought_together": [["C"], [], []],
        }
    )
    links = build_item_links(items)
    assert set(links["link_type"]) <= {"also_buy", "also_view", "bought_together"}
    assert len(links) >= 2


def test_neighbors_for_item_prefers_bought_together():
    links = pd.DataFrame(
        {
            "src_item_id": ["A", "A"],
            "dst_item_id": ["B", "C"],
            "link_type": ["bought_together", "also_view"],
        }
    )
    neighbors = neighbors_for_item(links, "A", limit=5)
    assert neighbors == ["B"]


@patch("recsys.data.download.download_multicategory_from_huggingface")
def test_prepare_dataset_writes_processed_artifacts(mock_download, tmp_path, monkeypatch):
    import pandas as pd

    from recsys.data.download import prepare_dataset

    items = pd.DataFrame(
        {
            "item_id": ["B001"],
            "title": ["Product"],
            "description": ["Desc"],
            "category": ["Books"],
            "main_category": ["Books"],
            "price": [9.99],
            "image_url": ["https://m.media-amazon.com/images/I/test.jpg"],
            "average_rating": [4.5],
            "rating_number": [10],
            "also_buy": [[]],
            "also_view": [[]],
            "bought_together": [[]],
        }
    )
    reviews = pd.DataFrame(
        {
            "user_id": ["u1"],
            "item_id": ["B001"],
            "rating": [5.0],
            "timestamp": [1_500_000_000],
            "verified_purchase": [True],
            "review_text": ["good"],
            "helpful_vote": [1],
            "main_category": ["Books"],
        }
    )
    mock_download.return_value = (reviews, items)

    processed_dir = tmp_path / "processed"
    raw_dir = tmp_path / "raw"

    class FakeCfg:
        seed = 42
        require_item_images = True
        target_interactions = 1000
        categories = [CategoryConfig(name="Books", config_suffix="Books", target_interactions=1)]
        splits = type("S", (), {"train_end": "2018-06-30", "val_end": "2018-12-31"})()
        paths = type("P", (), {"processed_dir": "processed", "raw_dir": "raw"})()

        def resolve_path(self, relative: str) -> Path:
            if relative == "processed":
                return processed_dir
            if relative == "raw":
                return raw_dir
            return tmp_path / relative

    monkeypatch.setattr("recsys.data.download.get_base_config", lambda: FakeCfg())
    paths = prepare_dataset()
    assert paths["interactions"].exists()
    assert paths["item_links"].exists()


@patch("recsys.data.download._stream_category_metadata")
@patch("recsys.data.download._stream_category_reviews")
def test_download_category_filters_to_image_backed_items(mock_reviews, mock_meta, tmp_path):
    mock_reviews.return_value = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "item_id": ["A", "B"],
            "rating": [5.0, 4.0],
            "timestamp": [1_500_000_000, 1_500_000_100],
            "verified_purchase": [True, False],
            "review_text": ["good", "ok"],
            "helpful_vote": [1, 0],
            "main_category": ["Books", "Books"],
        }
    )
    mock_meta.return_value = pd.DataFrame(
        {
            "item_id": ["A", "B"],
            "title": ["Book A", "Book B"],
            "description": ["desc", "desc"],
            "category": ["Books", "Books"],
            "main_category": ["Books", "Books"],
            "price": [9.99, 12.99],
            "image_url": ["https://m.media-amazon.com/images/I/a.jpg", ""],
            "average_rating": [4.5, 4.0],
            "rating_number": [10, 5],
            "also_buy": [[], []],
            "also_view": [[], []],
            "bought_together": [[], []],
        }
    )

    category = CategoryConfig(name="Books", config_suffix="Books", target_interactions=1)
    reviews, items = download_category_from_huggingface(
        tmp_path,
        category,
        target_interactions=1,
        seed=42,
        require_images=True,
    )
    assert set(reviews["item_id"]) == {"A"}
    assert set(items["item_id"]) == {"A"}
