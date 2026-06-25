from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from app.api.main import app
from app.storage.database import build_engine, build_session_factory, session_scope
from app.storage.upserts import upsert_raw_bars


def _api_route_bar_rows() -> list[dict]:
    start = datetime(2027, 1, 3, tzinfo=UTC)
    rows = []
    for index in range(80):
        event_time = start + timedelta(minutes=index)
        base = 110 + math.cos(index / 5) * 2
        close = base + 8 if index >= 76 else base
        rows.append(
            {
                "symbol": "BTCUSDT",
                "timeframe": "1m",
                "event_time": event_time,
                "fetch_time": event_time,
                "open": close - 0.1,
                "high": close + 0.7,
                "low": close - 0.7,
                "close": close,
                "volume": 25 + index,
                "source": "api-route-test-v2",
                "metadata_json": "{}",
            }
        )
    return rows


def test_features_build_can_run_research_pipeline_and_latest_signals_reads_db() -> None:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        upsert_raw_bars(session, _api_route_bar_rows())

    client = TestClient(app)
    response = client.post(
        "/features/build",
        params={
            "symbol": "BTCUSDT",
            "timeframe": "1m",
            "source": "api-route-test-v2",
            "run_research_pipeline": "true",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["dataset_rows"] == 80
    assert payload["decision"] in {"long", "short", "no_trade"}

    latest_response = client.get("/signals/latest", params={"symbol": "BTCUSDT", "limit": 5})
    assert latest_response.status_code == 200
    latest_payload = latest_response.json()
    assert any(signal["signal_id"] == payload["signal_id"] for signal in latest_payload["signals"])

    explain_response = client.get(f"/signals/{payload['signal_id']}/explain")
    assert explain_response.status_code == 200
    explanation = explain_response.json()
    assert explanation["signal_id"] == payload["signal_id"]
    assert explanation["audit_id"] == payload["audit_id"]
    assert explanation["decision"] == payload["decision"]
    assert "feature_zscore_close_20" in explanation["feature_snapshot"]
    assert "risk_checks" in explanation
    assert "cost_assumptions" in explanation


def test_signal_explain_returns_404_for_missing_signal() -> None:
    client = TestClient(app)
    response = client.get("/signals/does-not-exist/explain")
    assert response.status_code == 404
