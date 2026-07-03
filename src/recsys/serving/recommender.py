"""Live recommendation service wiring the full funnel."""

from __future__ import annotations

from dataclasses import dataclass, field

from recsys.env import configure_runtime_env

configure_runtime_env()

import numpy as np
import pandas as pd
import torch

from recsys.config import get_base_config, load_yaml
from recsys.data.item_links import load_item_links, neighbors_for_item
from recsys.data.splits import load_splits
from recsys.catalog import item_image_url, item_to_card
from recsys.features.embeddings import mean_embedding_for_items
from recsys.features.item_features import build_item_feature_matrix
from recsys.ranking.prerank import prerank_candidates
from recsys.reranking.calibration import ScoreCalibrator
from recsys.reranking.diversity import maximal_marginal_relevance
from recsys.retrieval.ann_index import ANNIndex
from recsys.retrieval.two_tower import TwoTowerConfig, TwoTowerModel
from recsys.serving.pipeline import ServingPipeline, StageLatency
from recsys.utils import get_device


@dataclass
class RecommendationResult:
    user_id: str
    slate: list[dict]
    latency: StageLatency
    within_budget: dict[str, bool]
    user_history: list[str] = field(default_factory=list)
    context_item_id: str | None = None


class _Retriever:
    def __init__(self, service: RecommendationService) -> None:
        self._service = service

    def retrieve(self, user_id: str, context: dict) -> list[str]:
        return self._service._retrieve(user_id)


class _Preranker:
    def __init__(self, service: RecommendationService) -> None:
        self._service = service

    def prerank(self, candidates: list[str], context: dict) -> list[str]:
        return self._service._prerank(candidates)


class _Ranker:
    def __init__(self, service: RecommendationService) -> None:
        self._service = service

    def rank(self, candidates: list[str], context: dict) -> list[str]:
        return self._service._rank(candidates, context.get("user_id"))


class _Reranker:
    def __init__(self, service: RecommendationService) -> None:
        self._service = service

    def rerank(self, candidates: list[str], context: dict) -> list[str]:
        return self._service._rerank(candidates, context.get("slate_size", 10))


class RecommendationService:
    def __init__(self) -> None:
        self.cfg = get_base_config()
        self.retrieval_cfg = load_yaml("retrieval.yaml")
        self.ranking_cfg = load_yaml("ranking.yaml")
        self.sim_cfg = load_yaml("simulator.yaml")
        self.device = get_device()
        self._loaded = False
        self.interactions: pd.DataFrame | None = None
        self.items: pd.DataFrame | None = None
        self.item_features: pd.DataFrame | None = None
        self.user_history: dict[str, list[str]] = {}
        self.item_id_to_idx: dict[str, int] = {}
        self.item_features_np: np.ndarray | None = None
        self.retrieval_model: TwoTowerModel | None = None
        self.ann_index: ANNIndex | None = None
        self.lgbm: object | None = None
        self.calibrator: ScoreCalibrator | None = None
        self.item_embeddings: dict[str, np.ndarray] = {}
        self.item_links: pd.DataFrame | None = None
        self._last_scores: dict[str, float] = {}
        self.pipeline: ServingPipeline | None = None
        self._context_item_id: str | None = None

    def load(self, force: bool = False) -> None:
        if self._loaded and not force:
            return
        processed = self.cfg.resolve_path(self.cfg.paths.processed_dir)
        features_dir = self.cfg.resolve_path(self.cfg.paths.features_dir)
        if not (processed / "interactions.parquet").exists():
            raise FileNotFoundError("Dataset not prepared. Run the data pipeline first.")

        self.interactions, self.items, _, _, _ = load_splits(processed)
        self.item_links = load_item_links(processed)
        if (features_dir / "item_features_batch.parquet").exists():
            self.item_features = pd.read_parquet(features_dir / "item_features_batch.parquet")
        else:
            self.item_features = pd.DataFrame({"item_id": self.items["item_id"]})

        self.user_history = {}
        for row in self.interactions.sort_values("timestamp").itertuples():
            self.user_history.setdefault(row.user_id, []).append(row.item_id)

        item_ids = self.items["item_id"].tolist()
        self.item_id_to_idx = {item_id: i + 1 for i, item_id in enumerate(item_ids)}
        self._build_item_features_np()
        self._load_retrieval()
        self._load_ranker()
        self._build_item_embeddings()
        self.pipeline = ServingPipeline(
            _Retriever(self),
            _Preranker(self),
            _Ranker(self),
            _Reranker(self),
        )
        self._loaded = True

    def _build_item_features_np(self) -> None:
        self.item_features_np = build_item_feature_matrix(self.items, self.item_features)

    def _load_retrieval(self) -> None:
        model_dir = self.cfg.resolve_path(self.cfg.paths.models_dir) / "retrieval"
        model_cfg = TwoTowerConfig(
            num_users=self.interactions["user_id"].nunique() + 1,
            num_items=len(self.item_id_to_idx) + 1,
            embedding_dim=self.retrieval_cfg["model"]["embedding_dim"],
            user_hidden_dim=self.retrieval_cfg["model"]["user_hidden_dim"],
            item_hidden_dim=self.retrieval_cfg["model"]["item_hidden_dim"],
            history_length=self.retrieval_cfg["model"]["history_length"],
            item_feature_dim=self.item_features_np.shape[1],
        )
        self.retrieval_model = TwoTowerModel(model_cfg).to(self.device)
        state_path = model_dir / "two_tower.pt"
        if state_path.exists():
            self.retrieval_model.load_state_dict(
                torch.load(state_path, map_location=self.device, weights_only=True)
            )

        self.ann_index = ANNIndex(self.retrieval_cfg["model"]["embedding_dim"])
        index_path = model_dir / "item_index.faiss"
        if index_path.exists():
            self.ann_index.load(index_path)
        else:
            rng = np.random.default_rng(self.cfg.seed)
            emb = rng.normal(
                size=(len(self.items), self.retrieval_cfg["model"]["embedding_dim"])
            ).astype(np.float32)
            self.ann_index.build(emb, self.items["item_id"].tolist())

    def _load_ranker(self) -> None:
        from recsys.ranking.lgbm_ranker import LightGBMRanker

        model_dir = self.cfg.resolve_path(self.cfg.paths.models_dir) / "ranking"
        self.lgbm = LightGBMRanker(self.ranking_cfg.get("lgbm"))
        lgbm_path = model_dir / "lgbm_ranker.txt"
        if lgbm_path.exists():
            self.lgbm.load(lgbm_path)
        self.calibrator = ScoreCalibrator(
            self.sim_cfg["reranking"].get("calibration_method", "isotonic")
        )

    def _build_item_embeddings(self) -> None:
        if self.retrieval_model is None or self.item_features_np is None:
            return
        item_ids = self.items["item_id"].tolist()
        indices = torch.tensor(
            [self.item_id_to_idx[i] for i in item_ids],
            device=self.device,
        )
        feats = torch.tensor(self.item_features_np, dtype=torch.float32, device=self.device)
        self.retrieval_model.eval()
        with torch.no_grad():
            emb = self.retrieval_model.encode_items(indices, feats).cpu().numpy()
        self.item_embeddings = {item_id: emb[i] for i, item_id in enumerate(item_ids)}

    def _retrieve(self, user_id: str) -> list[str]:
        history_length = self.retrieval_cfg["model"]["history_length"]
        top_k = self.retrieval_cfg["ann"]["top_k"]
        hist_ids = self.user_history.get(user_id, [])[-history_length:]
        if self._context_item_id and self._context_item_id in self.item_id_to_idx:
            hist_ids = (hist_ids + [self._context_item_id])[-history_length:]

        if hist_ids and self.retrieval_model and self.ann_index:
            hist_idx = [self.item_id_to_idx[i] for i in hist_ids if i in self.item_id_to_idx]
            if hist_idx:
                hist_tensor = torch.zeros(1, history_length, dtype=torch.long, device=self.device)
                hist_tensor[0, -len(hist_idx) :] = torch.tensor(hist_idx, device=self.device)
                self.retrieval_model.eval()
                with torch.no_grad():
                    user_emb = self.retrieval_model.encode_users(hist_tensor).cpu().numpy()
                results, scores = self.ann_index.search(user_emb, top_k=top_k)
                candidate_ids = list(results[0])
                self._last_scores = {
                    item_id: float(scores[0][i]) for i, item_id in enumerate(candidate_ids)
                }

                if self._context_item_id and self.item_links is not None:
                    linked = neighbors_for_item(self.item_links, self._context_item_id, limit=10)
                    max_score = max(self._last_scores.values(), default=0.5)
                    for neighbor in linked:
                        if neighbor not in candidate_ids:
                            candidate_ids.insert(0, neighbor)
                            self._last_scores[neighbor] = max_score + 0.05
                return candidate_ids

        fallback = self.items["item_id"].head(top_k).tolist()
        self._last_scores = {item_id: 0.1 for item_id in fallback}
        return fallback

    def _prerank(self, candidates: list[str]) -> list[str]:
        return prerank_candidates(
            {c: self._last_scores.get(c, 0.0) for c in candidates},
            self.item_features,
            top_k=self.ranking_cfg["prerank"]["top_k"],
        )

    def _rank(self, candidates: list[str], user_id: str | None = None) -> list[str]:
        from recsys.ranking.lgbm_ranker import build_ranking_features

        retrieval_scores = {c: self._last_scores.get(c, 0.0) for c in candidates}
        deep_scores = {c: retrieval_scores[c] * 0.8 for c in candidates}
        history_ids = self.user_history.get(user_id or "", [])[-10:]
        user_emb = mean_embedding_for_items(history_ids, self.items, embedding_df=self.item_features)
        features, valid_ids = build_ranking_features(
            candidates,
            retrieval_scores,
            deep_scores,
            self.item_features,
            user_content_embedding=user_emb,
        )
        if len(valid_ids) == 0:
            return candidates

        if self.lgbm and self.lgbm.model is not None:
            scores = self.lgbm.predict(features)
        else:
            scores = np.array([retrieval_scores[i] for i in valid_ids])

        if self.calibrator and len(scores) > 5:
            labels = (scores > np.median(scores)).astype(float)
            self.calibrator.fit(scores, labels)
            scores = self.calibrator.transform(scores)

        ranked = sorted(zip(valid_ids, scores, strict=False), key=lambda x: x[1], reverse=True)
        self._last_scores = {item_id: float(score) for item_id, score in ranked}
        return [item_id for item_id, _ in ranked]

    def _rerank(self, candidates: list[str], slate_size: int) -> list[str]:
        lambda_param = self.sim_cfg["reranking"]["mmr_lambda"]
        exclude = {self._context_item_id} if self._context_item_id else set()
        filtered = [c for c in candidates if c not in exclude]
        return maximal_marginal_relevance(
            filtered[:20],
            self._last_scores,
            self.item_embeddings,
            top_k=slate_size,
            lambda_param=lambda_param,
        )

    def recommend(
        self,
        user_id: str,
        slate_size: int = 10,
        context_item_id: str | None = None,
    ) -> RecommendationResult:
        self._context_item_id = context_item_id
        self.load()
        slate_ids, latency = self.pipeline.recommend(
            user_id,
            {"slate_size": slate_size, "user_id": user_id},
        )
        self._context_item_id = None
        items_lookup = self.items.set_index("item_id")
        slate = []
        for position, item_id in enumerate(slate_ids):
            if item_id in items_lookup.index:
                card = item_to_card(items_lookup.loc[item_id], item_id=item_id)
                slate.append(
                    {
                        **card,
                        "score": round(self._last_scores.get(item_id, 0.0), 4),
                        "position": position,
                    }
                )
            else:
                slate.append(
                    {
                        "item_id": item_id,
                        "title": item_id,
                        "category": "Unknown",
                        "price": None,
                        "score": round(self._last_scores.get(item_id, 0.0), 4),
                        "position": position,
                        "image_url": item_image_url(item_id, "Unknown"),
                    }
                )
        return RecommendationResult(
            user_id=user_id,
            slate=slate,
            latency=latency,
            within_budget=self.pipeline.within_budget(latency),
            user_history=self.user_history.get(user_id, [])[-10:],
            context_item_id=context_item_id,
        )

    def list_users(self, limit: int = 50) -> list[dict]:
        from recsys.api.services import list_users_from_dataset

        return list_users_from_dataset(limit=limit)

    def invalidate(self) -> None:
        self._loaded = False


_service: RecommendationService | None = None


def get_recommendation_service() -> RecommendationService:
    global _service
    if _service is None:
        _service = RecommendationService()
    return _service
