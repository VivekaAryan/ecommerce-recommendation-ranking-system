from recsys.config import get_base_config


def test_base_config_loads():
    cfg = get_base_config()
    assert cfg.category == "Electronics"
    assert cfg.target_interactions == 750_000
    assert cfg.splits.train_end == "2018-06-30"
