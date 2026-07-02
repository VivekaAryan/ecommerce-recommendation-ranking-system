from recsys.data.download import generate_synthetic_dataset
from recsys.simulator.logging import PropensityLogger
from recsys.simulator.runner import build_default_runner


def test_simulator_runs(tmp_path):
    reviews, items = generate_synthetic_dataset(num_interactions=200, num_items=30, num_users=20)
    runner = build_default_runner(items, reviews, tmp_path)
    logs = runner.run(num_days=1, sessions_per_day=10)
    assert len(logs) > 0
    assert "propensity" in logs.columns


def test_propensity_logger_flush(tmp_path):
    logger = PropensityLogger(tmp_path)
    assert logger.to_dataframe().empty
    path = logger.flush("test.parquet")
    assert path.exists()
