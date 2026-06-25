from __future__ import annotations

import tempfile

from fastapi.testclient import TestClient
from sqlalchemy import text

from app.api.main import app
from app.pipelines.backtest_pipeline import (
    run_backtest_pipeline_from_postgres,
    run_ml_walk_forward_backtest_from_postgres,
    run_strategy_comparison_from_postgres,
)
from app.pipelines.demo_seed import seed_demo_data
from app.pipelines.training_pipeline import run_training_pipeline_from_postgres
from app.storage.database import build_engine, build_session_factory
from app.storage.repositories import PostgresModelRepository


def test_backtest_pipeline_persists_run_and_api_can_fetch_it() -> None:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    seed_demo_data(
        session_factory=session_factory,
        symbols=["BTCUSDT"],
        rows=120,
        source="backtest-pipeline-test",
    )

    result = run_backtest_pipeline_from_postgres(
        session_factory=session_factory,
        symbol="BTCUSDT",
        timeframe="1m",
        source="backtest-pipeline-test",
    )

    client = TestClient(app)
    response = client.get(f"/backtests/{result.run_id}")
    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == result.run_id
    assert payload["metrics"]["trade_count"] == result.metrics["trade_count"]


def test_backtest_run_api_executes_pipeline() -> None:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    seed_demo_data(
        session_factory=session_factory,
        symbols=["ETHUSDT"],
        rows=120,
        source="backtest-api-test",
    )

    client = TestClient(app)
    response = client.post(
        "/backtest/run",
        params={
            "symbol": "ETHUSDT",
            "timeframe": "1m",
            "source": "backtest-api-test",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["dataset_rows"] == 120
    assert "total_return" in payload["metrics"]

    ui_response = client.get("/ui/backtest-runs")
    assert ui_response.status_code == 200
    assert any(run["run_id"] == payload["run_id"] for run in ui_response.json()["runs"])


def test_backtest_run_api_accepts_risk_exposure_params() -> None:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    seed_demo_data(
        session_factory=session_factory,
        symbols=["BTCUSDT"],
        rows=120,
        source="backtest-api-risk-params",
    )

    client = TestClient(app)
    response = client.post(
        "/backtest/run",
        params={
            "symbol": "BTCUSDT",
            "timeframe": "1m",
            "source": "backtest-api-risk-params",
            "max_concurrent_positions": 2,
            "max_positions_per_symbol": 1,
            "cooldown_bars_after_stop": 2,
            "allow_opposite_side_same_symbol": False,
            "apply_funding_schedule": True,
            "funding_rate": 0.0005,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "signals_seen" in payload["metrics"]
    assert "signals_skipped_exposure" in payload["metrics"]
    assert "funding_events_total" in payload["metrics"]


def test_backtest_walk_forward_api_returns_fold_summary() -> None:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    seed_demo_data(
        session_factory=session_factory,
        symbols=["BTCUSDT"],
        rows=220,
        source="backtest-wf-api",
    )

    client = TestClient(app)
    response = client.post(
        "/backtest/walk-forward",
        params={
            "symbol": "BTCUSDT",
            "timeframe": "1m",
            "source": "backtest-wf-api",
            "n_folds": 3,
            "min_train_bars": 120,
            "test_bars": 20,
            "apply_funding_schedule": True,
            "funding_rate": 0.0005,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["fold_rows"] == 3
    assert "wf_total_trades" in payload["aggregate_metrics"]
    assert len(payload["folds"]) == 3


def test_backtest_walk_forward_api_returns_400_for_invalid_window() -> None:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    seed_demo_data(
        session_factory=session_factory,
        symbols=["BTCUSDT"],
        rows=120,
        source="backtest-wf-api-err",
    )

    client = TestClient(app)
    response = client.post(
        "/backtest/walk-forward",
        params={
            "symbol": "BTCUSDT",
            "timeframe": "1m",
            "source": "backtest-wf-api-err",
            "n_folds": 3,
            "min_train_bars": 100,
            "test_bars": 20,
        },
    )
    assert response.status_code == 400
    assert "Not enough bars for walk-forward" in response.json()["detail"]


def test_ml_walk_forward_pipeline_returns_aggregate_metrics() -> None:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    source = "ml-wf-pipeline"
    seed_demo_data(
        session_factory=session_factory,
        symbols=["BTCUSDT"],
        rows=240,
        source=source,
    )

    with tempfile.TemporaryDirectory() as tmp:
        trained = run_training_pipeline_from_postgres(
            session_factory=session_factory,
            symbol="BTCUSDT",
            timeframe="1m",
            source=source,
            feature_set_version="v1",
            save_artifact=True,
            artifact_base_dir=tmp,
        )
        result = run_ml_walk_forward_backtest_from_postgres(
            session_factory=session_factory,
            model_run_id=trained.model_run_id,
            symbol="BTCUSDT",
            timeframe="1m",
            feature_set_version="v1",
            source=source,
            artifact_base_dir=tmp,
            n_folds=3,
            min_train_bars=120,
            test_bars=20,
        )

    assert result.model_run_id == trained.model_run_id
    assert result.signal_count > 0
    assert "wf_total_trades" in result.aggregate_metrics
    assert len(result.folds) == 3


def test_backtest_ml_walk_forward_api_smoke() -> None:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    source = "ml-wf-api"
    seed_demo_data(
        session_factory=session_factory,
        symbols=["ETHUSDT"],
        rows=240,
        source=source,
    )
    trained = run_training_pipeline_from_postgres(
        session_factory=session_factory,
        symbol="ETHUSDT",
        timeframe="1m",
        source=source,
        feature_set_version="v1",
        save_artifact=True,
    )

    client = TestClient(app)
    response = client.post(
        "/backtest/ml-walk-forward",
        params={
            "model_run_id": trained.model_run_id,
            "symbol": "ETHUSDT",
            "timeframe": "1m",
            "feature_set_version": "v1",
            "source": source,
            "n_folds": 3,
            "min_train_bars": 120,
            "test_bars": 20,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_run_id"] == trained.model_run_id
    assert "aggregate_metrics" in payload


def test_strategy_comparison_pipeline_includes_rankings() -> None:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    source = "strategy-cmp-pipeline"
    seed_demo_data(
        session_factory=session_factory,
        symbols=["BTCUSDT"],
        rows=260,
        source=source,
    )
    trained = run_training_pipeline_from_postgres(
        session_factory=session_factory,
        symbol="BTCUSDT",
        timeframe="1m",
        source=source,
        feature_set_version="v1",
        save_artifact=True,
    )
    result = run_strategy_comparison_from_postgres(
        session_factory=session_factory,
        model_run_id=trained.model_run_id,
        symbol="BTCUSDT",
        timeframe="1m",
        feature_set_version="v1",
        source=source,
        n_folds=3,
        min_train_bars=120,
        test_bars=20,
    )
    assert "ml_advisor" in result.compared_strategies
    assert "mean_reversion" in result.compared_strategies
    assert len(result.comparison_rows) >= 2
    assert "best_strategy_by_total_return" in result.summary
    assert "ml_vs_best_rule_total_return_delta" in result.summary
    assert all("rank_by_total_return" in row for row in result.comparison_rows)
    assert "quality_gates" in result.to_dict()
    assert "comparison_config" in result.to_dict()


def test_strategy_comparison_api_smoke() -> None:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    source = "strategy-cmp-api"
    seed_demo_data(
        session_factory=session_factory,
        symbols=["ETHUSDT"],
        rows=260,
        source=source,
    )
    trained = run_training_pipeline_from_postgres(
        session_factory=session_factory,
        symbol="ETHUSDT",
        timeframe="1m",
        source=source,
        feature_set_version="v1",
        save_artifact=True,
    )

    client = TestClient(app)
    response = client.post(
        "/backtest/strategy-comparison",
        params={
            "model_run_id": trained.model_run_id,
            "symbol": "ETHUSDT",
            "timeframe": "1m",
            "feature_set_version": "v1",
            "source": source,
            "n_folds": 3,
            "min_train_bars": 120,
            "test_bars": 20,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_run_id"] == trained.model_run_id
    assert "comparison_rows" in payload
    assert "summary" in payload
    assert "quality_gates" in payload
    assert "quality_gate_passed" in payload
    assert "quality_gate_warnings" in payload
    assert "comparison_config" in payload

    response_second = client.post(
        "/backtest/strategy-comparison",
        params={
            "model_run_id": trained.model_run_id,
            "symbol": "ETHUSDT",
            "timeframe": "1m",
            "feature_set_version": "v1",
            "source": source,
            "n_folds": 3,
            "min_train_bars": 120,
            "test_bars": 20,
        },
    )
    assert response_second.status_code == 200

    with session_factory() as session:
        snapshot = PostgresModelRepository(session).latest_strategy_comparison_snapshot(
            trained.model_run_id
        )
        count = session.execute(
            text("""
                select count(*)
                from strategy_comparison_snapshots
                where model_run_id = :model_run_id
            """),
            {"model_run_id": trained.model_run_id},
        ).scalar_one()

    assert snapshot is not None
    assert snapshot["model_run_id"] == trained.model_run_id
    assert snapshot["comparison_json"]["model_run_id"] == trained.model_run_id
    assert snapshot["comparison_json"]["symbol"] == "ETHUSDT"
    assert count == 1
