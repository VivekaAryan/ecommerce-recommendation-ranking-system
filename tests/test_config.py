from recsys.config import get_base_config


def test_base_config_loads():
    cfg = get_base_config()
    assert cfg.category == "Electronics"
    assert cfg.target_interactions == 175_000
    assert cfg.categories[0].target_interactions == 25_000
    assert cfg.require_item_images is True
    assert len(cfg.categories) == 7
    assert cfg.categories[0].name == "Appliances"
    assert cfg.splits.train_end == "2018-06-30"
