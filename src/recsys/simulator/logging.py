"""Propensity-aware interaction logging."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from recsys.simulator.policy import InteractionOutcome
from recsys.utils import ensure_dir


class PropensityLogger:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = ensure_dir(output_dir)
        self.rows: list[dict] = []

    def log(self, outcome: InteractionOutcome, timestamp: datetime | None = None) -> None:
        row = asdict(outcome)
        row["timestamp"] = (timestamp or datetime.utcnow()).isoformat()
        self.rows.append(row)

    def log_many(self, outcomes: list[InteractionOutcome], timestamp: datetime | None = None) -> None:
        for outcome in outcomes:
            self.log(outcome, timestamp=timestamp)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def flush(self, filename: str = "simulator_logs.parquet") -> Path:
        path = self.output_dir / filename
        self.to_dataframe().to_parquet(path, index=False)
        return path

    @staticmethod
    def load(path: Path) -> pd.DataFrame:
        return pd.read_parquet(path)
