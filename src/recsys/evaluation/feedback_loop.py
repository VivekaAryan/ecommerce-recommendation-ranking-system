"""Long-run feedback loop simulation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from recsys.evaluation.offline import catalog_coverage, gini_coefficient


@dataclass
class FeedbackLoopMetrics:
    cycle: int
    catalog_coverage: float
    gini: float
    num_interactions: int


def measure_feedback_loop(
    logs_per_cycle: list[pd.DataFrame],
    catalog_size: int,
) -> list[FeedbackLoopMetrics]:
    metrics: list[FeedbackLoopMetrics] = []
    for cycle, logs in enumerate(logs_per_cycle, start=1):
        counts = logs["item_id"].value_counts().to_numpy()
        metrics.append(
            FeedbackLoopMetrics(
                cycle=cycle,
                catalog_coverage=catalog_coverage(logs, catalog_size),
                gini=gini_coefficient(counts),
                num_interactions=len(logs),
            )
        )
    return metrics
