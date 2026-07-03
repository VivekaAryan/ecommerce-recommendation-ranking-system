"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"


def load_yaml(name: str) -> dict[str, Any]:
    path = CONFIGS_DIR / name
    with path.open() as f:
        return yaml.safe_load(f)


class SplitConfig(BaseModel):
    train_end: str
    val_end: str


class PathsConfig(BaseModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    features_dir: str = "data/features"
    simulator_logs_dir: str = "data/simulator_logs"
    models_dir: str = "models"


class MLflowConfig(BaseModel):
    tracking_uri: str = "sqlite:///data/mlflow.db"
    experiment_name: str = "recsys-platform"


class CategoryConfig(BaseModel):
    name: str
    config_suffix: str
    target_interactions: int | None = None


class BaseConfig(BaseModel):
    seed: int = 42
    data_dir: str = "data"
    category: str = "Electronics"
    target_interactions: int = 175_000
    require_item_images: bool = True
    categories: list[CategoryConfig] = Field(default_factory=list)
    splits: SplitConfig
    paths: PathsConfig = Field(default_factory=PathsConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)

    def resolve_path(self, relative: str) -> Path:
        return PROJECT_ROOT / relative


def get_base_config() -> BaseConfig:
    raw = load_yaml("base.yaml")
    return BaseConfig(**raw)


def merge_configs(*names: str) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for name in names:
        merged.update(load_yaml(name))
    return merged
