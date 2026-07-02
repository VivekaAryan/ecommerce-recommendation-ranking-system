import numpy as np
import torch

from recsys.retrieval.ann_index import ANNIndex
from recsys.retrieval.embedding_versioning import apply_alignment, procrustes_align
from recsys.retrieval.two_tower import TwoTowerConfig, TwoTowerModel, logq_corrected_loss


def test_two_tower_forward():
    cfg = TwoTowerConfig(num_users=10, num_items=100, embedding_dim=16, item_feature_dim=4)
    model = TwoTowerModel(cfg)
    history = torch.randint(1, 100, (2, 10))
    items = torch.tensor([1, 2])
    feats = torch.randn(2, 4)
    scores = model(history, items, feats)
    assert scores.shape == (2,)


def test_logq_loss():
    logits = torch.randn(4, 4)
    pop = torch.tensor([1.0, 2.0, 3.0, 4.0])
    loss = logq_corrected_loss(logits, pop)
    assert loss.ndim == 0


def test_ann_index_search():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(20, 8)).astype(np.float32)
    ids = [f"i{i}" for i in range(20)]
    index = ANNIndex(8, index_type="flat")
    index.build(emb, ids)
    results, scores = index.search(emb[:1], top_k=5)
    assert len(results[0]) == 5


def test_embedding_alignment():
    old = np.random.randn(10, 4)
    new = old + 0.01 * np.random.randn(10, 4)
    transform = procrustes_align(old, new)
    aligned = apply_alignment(new, transform)
    assert aligned.shape == new.shape
