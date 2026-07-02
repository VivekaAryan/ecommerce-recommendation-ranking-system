"""In-memory background job manager for pipeline tasks."""

from __future__ import annotations

import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class Job:
    id: str
    task: str
    status: str = "pending"
    message: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None


class JobManager:
    def __init__(self, max_workers: int = 1) -> None:
        self._jobs: dict[str, Job] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, task: str, fn: Callable[[], dict[str, Any]]) -> Job:
        job_id = str(uuid.uuid4())[:8]
        job = Job(id=job_id, task=task, status="running", message="Started")
        self._jobs[job_id] = job

        def _run() -> None:
            try:
                result = fn()
                job.status = "completed"
                job.result = result
                job.message = result.get("message", "Completed")
            except Exception as exc:
                job.status = "failed"
                job.error = str(exc)
                job.message = traceback.format_exc()
            finally:
                job.completed_at = datetime.now(UTC)

        self._executor.submit(_run)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 20) -> list[Job]:
        jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]
