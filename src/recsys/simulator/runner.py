"""Multi-day simulator orchestration."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from recsys.config import load_yaml
from recsys.simulator.environment import MarketplaceEnvironment
from recsys.simulator.logging import PropensityLogger
from recsys.simulator.policy import ServingPolicy

CandidateFn = Callable[[MarketplaceEnvironment, str], list[str]]
RetrainHook = Callable[[pd.DataFrame], None]


class SimulatorRunner:
    def __init__(
        self,
        env: MarketplaceEnvironment,
        policy: ServingPolicy,
        logger: PropensityLogger,
        slate_size: int = 10,
        candidate_fn: CandidateFn | None = None,
        retrain_hook: RetrainHook | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.env = env
        self.policy = policy
        self.logger = logger
        self.slate_size = slate_size
        self.candidate_fn = candidate_fn or self._default_candidates
        self.retrain_hook = retrain_hook
        self.rng = rng or np.random.default_rng(42)

    def _default_candidates(self, env: MarketplaceEnvironment, user_id: str) -> list[str]:
        item_ids = list(env.items.keys())
        sample_size = min(500, len(item_ids))
        return self.rng.choice(item_ids, size=sample_size, replace=False).tolist()

    def run_day(self, sessions: int, day_offset: int = 0) -> pd.DataFrame:
        user_ids = list(self.env.users.keys())
        timestamp = datetime.utcnow() + timedelta(days=day_offset)
        for _ in range(sessions):
            user_id = self.rng.choice(user_ids)
            candidates = self.candidate_fn(self.env, user_id)
            slate = self.policy.build_slate(self.env, user_id, candidates, self.slate_size)
            outcomes = self.policy.simulate_interactions(self.env, user_id, slate)
            self.logger.log_many(outcomes, timestamp=timestamp)
        return self.logger.to_dataframe()

    def run(self, num_days: int, sessions_per_day: int) -> pd.DataFrame:
        for day in range(num_days):
            self.run_day(sessions_per_day, day_offset=day)
            if self.retrain_hook is not None and (day + 1) % 7 == 0:
                self.retrain_hook(self.logger.to_dataframe())
            self.env.restock()
        return self.logger.to_dataframe()


def build_default_runner(
    items_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    log_dir,
    candidate_fn: CandidateFn | None = None,
) -> SimulatorRunner:
    sim_cfg = load_yaml("simulator.yaml")
    env_cfg = sim_cfg["environment"]
    policy_cfg = sim_cfg["policy"]

    env = MarketplaceEnvironment.from_dataframes(
        items_df,
        interactions_df,
        num_synthetic_users=env_cfg["num_synthetic_users"],
        default_inventory=env_cfg["default_inventory"],
        attention_decay_rate=env_cfg["attention_decay_rate"],
    )
    policy = ServingPolicy(
        position_bias=policy_cfg["position_bias"],
        click_temperature=policy_cfg["click_temperature"],
        purchase_given_click_prob=policy_cfg["purchase_given_click_prob"],
        policy_id=sim_cfg["simulation"]["policy_id"],
    )
    logger = PropensityLogger(log_dir)
    return SimulatorRunner(
        env=env,
        policy=policy,
        logger=logger,
        slate_size=policy_cfg["slate_size"],
        candidate_fn=candidate_fn,
    )
