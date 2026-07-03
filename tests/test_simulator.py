from recsys.simulator.runner import build_default_runner
from tests.fixtures.catalog import make_test_interactions, make_test_items


def test_simulator_runs(tmp_path):
    items = make_test_items(num_items=30)
    reviews = make_test_interactions(items, num_interactions=200, num_users=20)
    runner = build_default_runner(items, reviews, tmp_path)
    logs = runner.run(num_days=2, sessions_per_day=10)
    assert len(logs) > 0
    assert "clicked" in logs.columns
