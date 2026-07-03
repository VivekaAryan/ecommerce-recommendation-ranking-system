from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from recsys.api.app import app
from recsys.config import get_base_config
from tests.conftest import write_test_processed_data


def _ensure_test_dataset() -> None:
    cfg = get_base_config()
    processed = cfg.resolve_path(cfg.paths.processed_dir)
    if not (processed / "interactions.parquet").exists():
        write_test_processed_data(processed)


def test_health():
    client = TestClient(app)
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_status():
    client = TestClient(app)
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert "artifacts" in data
    assert "ready" in data


def test_list_users_with_dataset():
    _ensure_test_dataset()
    client = TestClient(app)
    res = client.get("/api/users?limit=10")
    assert res.status_code == 200
    users = res.json()
    assert len(users) > 0
    assert "user_id" in users[0]
    assert "interaction_count" in users[0]
    assert users[0]["interaction_count"] > 0


def test_list_users_no_dataset():
    cfg = get_base_config()
    interactions_path = cfg.resolve_path(cfg.paths.processed_dir) / "interactions.parquet"
    backup: Path | None = None
    if interactions_path.exists():
        backup = interactions_path.with_suffix(".parquet.bak")
        interactions_path.rename(backup)

    try:
        client = TestClient(app)
        res = client.get("/api/users")
        assert res.status_code == 404
        assert "Dataset not prepared" in res.json()["detail"]
    finally:
        if backup is not None and backup.exists():
            backup.rename(interactions_path)


def test_list_users_from_dataset_unit():
    from recsys.api.services import list_users_from_dataset

    with patch("recsys.api.services.pd.read_parquet") as mock_read:
        import pandas as pd

        mock_read.return_value = pd.DataFrame(
            {
                "user_id": ["user_a", "user_a", "user_b"],
                "item_id": ["i1", "i2", "i3"],
            }
        )
        with patch("recsys.api.services.get_base_config") as mock_cfg:
            mock_cfg.return_value.resolve_path.return_value = Path("/fake/processed")
            with patch.object(Path, "exists", return_value=True):
                users = list_users_from_dataset(limit=5)
    assert users == [
        {"user_id": "user_a", "interaction_count": 2},
        {"user_id": "user_b", "interaction_count": 1},
    ]


def test_catalog_endpoint():
    _ensure_test_dataset()
    client = TestClient(app)
    res = client.get("/api/catalog?limit=5")
    assert res.status_code == 200
    data = res.json()
    assert len(data["products"]) <= 5
    assert "image_url" in data["products"][0]
    assert len(data["categories"]) > 0


def test_recommend_with_context():
    _ensure_test_dataset()
    client = TestClient(app)
    users = client.get("/api/users?limit=1").json()
    catalog = client.get("/api/catalog?limit=1").json()
    uid = users[0]["user_id"]
    item_id = catalog["products"][0]["item_id"]
    res = client.post(
        "/api/recommend",
        json={"user_id": uid, "slate_size": 5, "context_item_id": item_id},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["context_item_id"] == item_id
    assert all(s["item_id"] != item_id for s in body["slate"])
