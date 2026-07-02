"""API request/response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class JobRequest(BaseModel):
    task: str
    synthetic: bool = True
    synthetic_size: int = 10_000
    num_days: int | None = None
    sessions_per_day: int | None = None


class JobResponse(BaseModel):
    id: str
    task: str
    status: str
    message: str = ""


class JobDetail(BaseModel):
    id: str
    task: str
    status: str
    message: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None


class ArtifactStatus(BaseModel):
    name: str
    exists: bool
    path: str
    detail: str = ""


class SystemStatus(BaseModel):
    ready: bool
    artifacts: list[ArtifactStatus]
    dataset: dict[str, Any] | None = None


class RecommendRequest(BaseModel):
    user_id: str
    slate_size: int = 10


class SlateItem(BaseModel):
    item_id: str
    title: str
    category: str
    price: float | None
    score: float
    position: int


class LatencyBreakdown(BaseModel):
    retrieval_ms: float
    prerank_ms: float
    ranking_ms: float
    reranking_ms: float
    total_ms: float
    within_budget: dict[str, bool]


class RecommendResponse(BaseModel):
    user_id: str
    slate: list[SlateItem]
    latency: LatencyBreakdown
    user_history: list[str] = Field(default_factory=list)


class UserInfo(BaseModel):
    user_id: str
    interaction_count: int


class MetricsResponse(BaseModel):
    skew: list[dict[str, Any]] | None = None
    ope: dict[str, float] | None = None
    offline: dict[str, float] | None = None
    simulator_summary: dict[str, Any] | None = None
