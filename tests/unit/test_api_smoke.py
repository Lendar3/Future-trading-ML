from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.main import app


def test_health_endpoint_returns_ok() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["live_trading"] == "disabled"
