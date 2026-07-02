from datetime import datetime

from recsys.data.download import generate_synthetic_dataset
from recsys.features.batch import build_batch_features
from recsys.features.online import OnlineFeaturePipeline, measure_feature_skew


def test_batch_and_online_features_differ():
    reviews, items = generate_synthetic_dataset(num_interactions=500, num_items=50, num_users=30)
    as_of = datetime.utcfromtimestamp(int(reviews["timestamp"].max()))
    batch_user, _ = build_batch_features(reviews, items, as_of=as_of)
    online = OnlineFeaturePipeline()
    online_user, _ = online.build_online_snapshot(reviews, items, as_of=as_of)
    skew = measure_feature_skew(batch_user, online_user)
    assert not skew.empty
