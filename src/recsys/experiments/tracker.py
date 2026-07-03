"""MLflow experiment tracking wrapper."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow

from recsys.config import PROJECT_ROOT, get_base_config


def _resolve_tracking_uri(uri: str) -> str:
    """Resolve sqlite tracking URIs relative to the project root."""
    if uri.startswith("sqlite:///"):
        db_path = uri[len("sqlite:///") :]
        if not Path(db_path).is_absolute():
            db_path = str((PROJECT_ROOT / db_path).resolve())
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_path}"
    return uri


class ExperimentTracker:
    def __init__(self, experiment_name: str | None = None, tracking_uri: str | None = None) -> None:
        cfg = get_base_config()
        mlflow.set_tracking_uri(_resolve_tracking_uri(tracking_uri or cfg.mlflow.tracking_uri))
        mlflow.set_experiment(experiment_name or cfg.mlflow.experiment_name)
        self._active_run = None

    @contextmanager
    def run(self, run_name: str, tags: dict[str, str] | None = None) -> Iterator[None]:
        with mlflow.start_run(run_name=run_name):
            self._active_run = True
            if tags:
                mlflow.set_tags(tags)
            yield
            self._active_run = None

    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str) -> None:
        mlflow.log_artifact(path)

    def log_dict(self, dictionary: dict[str, Any], filename: str) -> None:
        mlflow.log_dict(dictionary, filename)
