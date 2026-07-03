from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd

from recsys.features.batch import build_batch_features
from recsys.features.embeddings import EMBEDDING_DIM
from recsys.features.online import OnlineFeaturePipeline, measure_feature_skew
from tests.fixtures.catalog import make_test_interactions, make_test_items


def _fake_embedding_df(items: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = {"item_id": items["item_id"].values}
    for i in range(EMBEDDING_DIM):
        data[f"content_emb_{i}"] = rng.random(len(items))
    return pd.DataFrame(data)


@patch("recsys.features.batch.compute_content_embeddings")
def test_batch_and_online_features_differ(mock_embeddings):
    items = make_test_items(num_items=50)
    reviews = make_test_interactions(items, num_interactions=500, num_users=30)
    embedding_df = _fake_embedding_df(items)
    mock_embeddings.return_value = (embedding_df, np.zeros((len(items), EMBEDDING_DIM), dtype=np.float32))

    as_of = datetime.utcfromtimestamp(int(reviews["timestamp"].max()))
    batch_user, _ = build_batch_features(reviews, items, as_of=as_of)
    online = OnlineFeaturePipeline()
    online_user, _ = online.build_online_snapshot(reviews, items, as_of=as_of)
    skew = measure_feature_skew(batch_user, online_user)
    assert not skew.empty
