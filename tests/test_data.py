import pandas as pd

from recsys.data.download import generate_synthetic_dataset, prepare_dataset
from recsys.data.subsample import stratified_subsample


def test_generate_synthetic_dataset():
    reviews, items = generate_synthetic_dataset(num_interactions=100, num_items=20, num_users=10)
    assert len(reviews) > 0
    assert len(items) == 20


def test_stratified_subsample():
    df = pd.DataFrame({"timestamp": range(1000), "user_id": ["u1"] * 1000})
    sampled = stratified_subsample(df, target_size=100, seed=42)
    assert len(sampled) == 100


def test_prepare_dataset_synthetic():
    paths = prepare_dataset(use_synthetic=True, synthetic_size=500)
    assert paths["interactions"].exists()
