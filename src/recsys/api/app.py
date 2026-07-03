"""FastAPI application for the recsys testing dashboard."""

from __future__ import annotations

from pathlib import Path

from recsys.env import configure_runtime_env

configure_runtime_env()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from recsys.api.jobs import JobManager
from recsys.api.schemas import (
    CatalogResponse,
    JobDetail,
    JobRequest,
    JobResponse,
    LatencyBreakdown,
    MetricsResponse,
    ProductCard,
    RecommendRequest,
    RecommendResponse,
    SlateItem,
    SystemStatus,
    UserHistoryResponse,
    UserInfo,
)
from recsys.api.services import (
    TASK_HANDLERS,
    get_catalog,
    get_metrics,
    get_simulator_logs,
    get_system_status,
    get_user_history_products,
    list_users_from_dataset,
)

FRONTEND_DIST = Path(__file__).resolve().parents[3] / "frontend" / "dist"

app = FastAPI(
    title="Recsys Platform Tester",
    description="Dashboard for testing the recommendation and ranking pipeline",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

job_manager = JobManager(max_workers=1)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/status", response_model=SystemStatus)
def status() -> SystemStatus:
    data = get_system_status()
    return SystemStatus(**data)


@app.get("/api/users", response_model=list[UserInfo])
def list_users(limit: int = 50) -> list[UserInfo]:
    try:
        users = list_users_from_dataset(limit=limit)
        return [UserInfo(**u) for u in users]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/catalog", response_model=CatalogResponse)
def catalog(limit: int = 48, offset: int = 0, category: str | None = None) -> CatalogResponse:
    try:
        data = get_catalog(limit=limit, offset=offset, category=category)
        return CatalogResponse(
            products=[ProductCard(**p) for p in data["products"]],
            categories=data["categories"],
            total=data["total"],
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/users/{user_id}/history", response_model=UserHistoryResponse)
def user_history(user_id: str, limit: int = 12) -> UserHistoryResponse:
    try:
        data = get_user_history_products(user_id, limit=limit)
        return UserHistoryResponse(
            user_id=data["user_id"],
            products=[ProductCard(**p) for p in data["products"]],
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest) -> RecommendResponse:
    from recsys.serving.recommender import get_recommendation_service

    try:
        service = get_recommendation_service()
        result = service.recommend(
            request.user_id,
            slate_size=request.slate_size,
            context_item_id=request.context_item_id,
        )
        return RecommendResponse(
            user_id=result.user_id,
            slate=[SlateItem(**item) for item in result.slate],
            latency=LatencyBreakdown(
                retrieval_ms=round(result.latency.retrieval_ms, 2),
                prerank_ms=round(result.latency.prerank_ms, 2),
                ranking_ms=round(result.latency.ranking_ms, 2),
                reranking_ms=round(result.latency.reranking_ms, 2),
                total_ms=round(result.latency.total_ms, 2),
                within_budget=result.within_budget,
            ),
            user_history=result.user_history,
            context_item_id=result.context_item_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/jobs", response_model=JobResponse)
def create_job(request: JobRequest) -> JobResponse:
    handler = TASK_HANDLERS.get(request.task)
    if handler is None:
        raise HTTPException(status_code=400, detail=f"Unknown task: {request.task}")
    job = job_manager.submit(request.task, lambda: handler(request))
    return JobResponse(id=job.id, task=job.task, status=job.status, message=job.message)


@app.get("/api/jobs", response_model=list[JobDetail])
def list_jobs() -> list[JobDetail]:
    return [
        JobDetail(
            id=j.id,
            task=j.task,
            status=j.status,
            message=j.message,
            result=j.result,
            error=j.error,
        )
        for j in job_manager.list_jobs()
    ]


@app.get("/api/jobs/{job_id}", response_model=JobDetail)
def get_job(job_id: str) -> JobDetail:
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobDetail(
        id=job.id,
        task=job.task,
        status=job.status,
        message=job.message,
        result=job.result,
        error=job.error,
    )


@app.get("/api/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    data = get_metrics()
    return MetricsResponse(**data)


@app.get("/api/simulator/logs")
def simulator_logs(limit: int = 100, offset: int = 0) -> dict:
    return get_simulator_logs(limit=limit, offset=offset)


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
