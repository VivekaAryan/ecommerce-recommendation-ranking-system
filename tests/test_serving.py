from recsys.serving.pipeline import ServingPipeline, StageLatency


class StubRetriever:
    def retrieve(self, user_id, context):
        return ["a", "b", "c"]


class StubPreranker:
    def prerank(self, candidates, context):
        return candidates[:2]


class StubRanker:
    def rank(self, candidates, context):
        return candidates


class StubReranker:
    def rerank(self, candidates, context):
        return candidates


def test_serving_pipeline():
    pipeline = ServingPipeline(StubRetriever(), StubPreranker(), StubRanker(), StubReranker())
    slate, latency = pipeline.recommend("u1", {})
    assert len(slate) == 2
    assert isinstance(latency, StageLatency)
    checks = pipeline.within_budget(latency)
    assert "total" in checks
