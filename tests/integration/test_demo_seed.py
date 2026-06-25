from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.main import app
from app.pipelines.demo_seed import generate_demo_bar_rows, seed_demo_data
from app.storage.database import build_engine, build_session_factory


def test_demo_seed_pipeline_creates_signals_for_selected_symbols() -> None:
    engine = build_engine()
    session_factory = build_session_factory(engine)

    result = seed_demo_data(
        session_factory=session_factory,
        symbols=["BTCUSDT", "ETHUSDT"],
        rows=100,
        source="demo-seed-test",
    )

    assert result.bars_upserted == 200
    assert [item.signal_id for item in result.pipeline_results]
    assert {item.decision for item in result.pipeline_results}.issubset(
        {"long", "short", "no_trade"}
    )


def test_demo_seed_api_creates_explainable_signal() -> None:
    client = TestClient(app)
    response = client.post(
        "/demo/seed",
        params={
            "symbols": "BTCUSDT",
            "rows": 100,
            "source": "demo-seed-api-test",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["bars_upserted"] == 100
    signal_id = payload["pipeline_results"][0]["signal_id"]

    explain_response = client.get(f"/signals/{signal_id}/explain")
    assert explain_response.status_code == 200
    assert explain_response.json()["signal_id"] == signal_id


def test_generate_demo_bar_rows_uses_requested_shape() -> None:
    rows = generate_demo_bar_rows("BTCUSDT", rows=80, source="shape-test")
    assert len(rows) == 80
    assert rows[0]["symbol"] == "BTCUSDT"
    assert rows[0]["source"] == "shape-test"
