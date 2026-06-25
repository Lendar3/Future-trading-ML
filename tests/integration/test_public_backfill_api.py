from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient

import app.api.routes_data as routes_data
from app.api.main import app
from tests.integration.test_bybit_public_backfill_fixture import FixtureBybitClient


def test_public_backfill_api_uses_public_fixture_client(monkeypatch) -> None:
    monkeypatch.setattr(
        routes_data,
        "BybitPublicRestClient",
        lambda: FixtureBybitClient(source="bybit_api_fixture"),
    )
    client = TestClient(app)
    response = client.post(
        "/collect/public-backfill",
        params={
            "symbol": "BTCUSDT",
            "start": datetime(2026, 5, 8, tzinfo=UTC).isoformat(),
            "end": datetime(2026, 5, 8, 0, 5, tzinfo=UTC).isoformat(),
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["timeframe"] == "1m"
    assert payload["bybit_interval"] == "1"
    assert payload["regular_bars"] == 2
    assert payload["funding_points"] == 1
