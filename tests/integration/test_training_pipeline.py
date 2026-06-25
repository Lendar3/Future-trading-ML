from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.main import app
from app.pipelines.demo_seed import seed_demo_data
from app.pipelines.training_pipeline import run_training_pipeline_from_postgres
from app.storage.database import build_engine, build_session_factory


def test_training_pipeline_persists_model_run_and_api_lists_it() -> None:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    seed_demo_data(
        session_factory=session_factory,
        symbols=["BTCUSDT"],
        rows=120,
        source="training-pipeline-test",
    )

    result = run_training_pipeline_from_postgres(
        session_factory=session_factory,
        symbol="BTCUSDT",
        timeframe="1m",
        source="training-pipeline-test",
    )

    client = TestClient(app)
    response = client.get("/models/latest", params={"symbol": "BTCUSDT", "limit": 20})
    assert response.status_code == 200
    assert any(run["model_run_id"] == result.model_run_id for run in response.json()["model_runs"])


def test_train_lightgbm_api_executes_training_pipeline() -> None:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    seed_demo_data(
        session_factory=session_factory,
        symbols=["ETHUSDT"],
        rows=120,
        source="training-api-test",
    )

    client = TestClient(app)
    response = client.post(
        "/train/lightgbm",
        params={
            "symbol": "ETHUSDT",
            "timeframe": "1m",
            "source": "training-api-test",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["dataset_rows"] > 0
    assert payload["metrics"]["features"] > 0

    ui_response = client.get("/ui/training-runs")
    assert ui_response.status_code == 200
    assert any(run["model_run_id"] == payload["model_run_id"] for run in ui_response.json()["runs"])
