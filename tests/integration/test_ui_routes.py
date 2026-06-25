from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.main import app
from app.pipelines.demo_seed import seed_demo_data
from app.storage.database import build_engine, build_session_factory


def test_ui_routes_return_chart_data_feature_vector_and_signal_explanation() -> None:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    seed_result = seed_demo_data(
        session_factory=session_factory,
        symbols=["BTCUSDT"],
        rows=100,
        source="ui-route-test",
    )
    signal_id = seed_result.pipeline_results[0].signal_id

    client = TestClient(app)
    chart_response = client.get(
        "/ui/chart-data",
        params={"symbol": "BTCUSDT", "timeframe": "1m", "source": "ui-route-test"},
    )
    assert chart_response.status_code == 200
    chart_payload = chart_response.json()
    assert chart_payload["symbol"] == "BTCUSDT"
    assert len(chart_payload["candles"]) == 100
    assert len(chart_payload["volume"]) == 100

    feature_response = client.get("/ui/feature-vector/BTCUSDT")
    assert feature_response.status_code == 200
    assert "features" in feature_response.json()

    explanation_response = client.get(f"/ui/signal-explanation/{signal_id}")
    assert explanation_response.status_code == 200
    explanation = explanation_response.json()
    assert explanation["signal_id"] == signal_id
    assert "feature_zscore_close_20" in explanation["feature_snapshot"]
