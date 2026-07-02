import numpy as np
import torch

from recsys.evaluation.adversarial import (
    detect_engagement_velocity_anomalies,
    inject_synthetic_manipulation,
)
from recsys.evaluation.interference import naive_ab_test, switchback_test
from recsys.evaluation.offline import ndcg_at_k, recall_at_k
from recsys.evaluation.ope import ips_estimator
from recsys.ranking.multitask import MultiTaskRanker, MultiTaskWeights
from recsys.reranking.calibration import ScoreCalibrator
from recsys.reranking.diversity import intra_list_diversity, maximal_marginal_relevance


def test_offline_metrics():
    preds = ["a", "b", "c"]
    truth = {"a", "c"}
    assert ndcg_at_k(preds, truth, 3) > 0
    assert recall_at_k(preds, truth, 3) > 0


def test_ips_estimator():
    import pandas as pd

    logs = pd.DataFrame(
        {
            "clicked": [1, 0, 1, 0],
            "propensity": [0.5, 0.5, 0.25, 0.25],
        }
    )
    value = ips_estimator(logs)
    assert isinstance(value, float)


def test_multitask_combined_score():
    model = MultiTaskRanker(input_dim=10)
    x = torch.randn(3, 10)
    outputs = model(x)
    score = model.combined_score(outputs, MultiTaskWeights())
    assert score.shape == (3,)


def test_mmr_diversity():
    item_ids = ["a", "b", "c"]
    scores = {"a": 1.0, "b": 0.9, "c": 0.8}
    emb = {k: np.array([1.0, 0.0]) if k == "a" else np.array([0.0, 1.0]) for k in item_ids}
    slate = maximal_marginal_relevance(item_ids, scores, emb, top_k=2, lambda_param=0.5)
    assert len(slate) == 2
    assert intra_list_diversity(slate, emb) >= 0


def test_calibration():
    scores = np.linspace(0, 1, 100)
    labels = (scores > 0.5).astype(int)
    calibrator = ScoreCalibrator("isotonic")
    calibrator.fit(scores, labels)
    calibrated = calibrator.transform(scores)
    assert len(calibrated) == 100


def test_interference_experiments():
    import pandas as pd

    logs = pd.DataFrame(
        {
            "policy_id": ["p_control", "p_treatment", "p_control", "p_treatment"],
            "clicked": [0, 1, 0, 1],
            "timestamp": pd.date_range("2020-01-01", periods=4, freq="h"),
        }
    )
    naive = naive_ab_test(logs)
    switchback = switchback_test(logs)
    assert naive.design == "naive_ab"
    assert switchback.design == "switchback"


def test_adversarial_detection():
    import pandas as pd

    logs = pd.DataFrame(
        {
            "item_id": ["a"] * 10,
            "clicked": [1] * 10,
            "timestamp": pd.date_range("2020-01-01", periods=10, freq="h"),
        }
    )
    manipulated = inject_synthetic_manipulation(logs, ["a"])
    assert len(manipulated) > len(logs)
    anomalies = detect_engagement_velocity_anomalies(manipulated)
    assert isinstance(anomalies, pd.DataFrame)
