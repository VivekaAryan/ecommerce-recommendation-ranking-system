import pandas as pd

from recsys.data.subsample import stratified_subsample
from recsys.data.timestamps import normalize_timestamps, normalize_unix_timestamp
from tests.fixtures.catalog import make_test_interactions, make_test_items


def test_normalize_unix_timestamp_milliseconds():
    assert normalize_unix_timestamp(1_500_000_000_000) == 1_500_000_000
    assert normalize_unix_timestamp(873_931_557_000) == 873_931_557
    assert normalize_unix_timestamp(1_500_000_000) == 1_500_000_000


def test_normalize_timestamps_dataframe():
    df = pd.DataFrame({"timestamp": [1_500_000_000_000, 1_600_000_000_000]})
    normalized = normalize_timestamps(df)
    assert normalized["timestamp"].tolist() == [1_500_000_000, 1_600_000_000]


def test_make_test_items_schema():
    items = make_test_items(num_items=20)
    assert len(items) == 20
    assert "image_url" in items.columns
    assert items["image_url"].str.startswith("https://m.media-amazon.com").all()
    assert items["item_id"].str.startswith("B").all()


def test_stratified_subsample():
    df = pd.DataFrame({"timestamp": range(1000), "user_id": ["u1"] * 1000})
    sampled = stratified_subsample(df, target_size=100, seed=42)
    assert len(sampled) == 100


def test_make_test_interactions():
    items = make_test_items(num_items=20)
    reviews = make_test_interactions(items, num_interactions=100, num_users=10)
    assert len(reviews) > 0
    assert "verified_purchase" in reviews.columns
