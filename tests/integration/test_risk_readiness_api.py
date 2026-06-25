from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.main import app
from app.pipelines.demo_seed import seed_demo_data
from app.storage.database import build_engine, build_session_factory


def test_risk_state_and_readiness_report_are_api_backed() -> None:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    seed_demo_data(
        session_factory=session_factory,
        symbols=["BTCUSDT"],
        rows=100,
        source="risk-readiness-test",
    )
    client = TestClient(app)
    client.post(
        "/backtest/run",
        params={
            "symbol": "BTCUSDT",
            "timeframe": "1m",
            "source": "risk-readiness-test",
        },
    )
    client.post("/demo/reset")

    risk_response = client.get("/ui/risk-state")
    readiness_response = client.get("/ui/readiness-report")
    live_metrics_response = client.get("/metrics/live")

    assert risk_response.status_code == 200
    assert readiness_response.status_code == 200
    assert live_metrics_response.status_code == 200
    assert risk_response.json()["live_trading"] == "disabled"
    assert risk_response.json()["risk_limits"]["max_open_positions"] == 3
    readiness = readiness_response.json()
    assert readiness["minimum_history_collected"] is True
    assert readiness["backtest_completed"] is True
    assert readiness["live_trading_disabled"] is True
    assert live_metrics_response.json()["live_trading"] == "disabled"
