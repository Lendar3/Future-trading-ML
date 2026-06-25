from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import text

from app.storage.database import build_engine, build_session_factory, session_scope
from app.storage.models import SignalAuditRow
from app.storage.repositories import PostgresAuditRepository, PostgresModelRepository
from app.storage.upserts import upsert_raw_bars

pytestmark = pytest.mark.integration


def test_postgres_audit_repository_round_trip() -> None:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    row = SignalAuditRow(
        audit_id="audit-integration-1",
        signal_id="sig-integration-1",
        symbol="BTCUSDT",
        event_time=datetime(2026, 1, 1, tzinfo=UTC),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        strategy_name="mean_reversion",
        model_run_id=None,
        feature_set_version="v1",
        feature_snapshot_json={"feature_ret_1m": 0.01},
        rationale_json={"reason": "integration"},
        risk_checks_json={"status": "accepted"},
        cost_assumptions_json={"taker_fee_bp": 5.5},
        decision="long",
        decision_reason="integration smoke",
        system_mode="paper",
        code_version="local",
        config_hash="hash",
    )

    with session_scope(session_factory) as session:
        repo = PostgresAuditRepository(session)
        repo.upsert(row)

    with session_scope(session_factory) as session:
        repo = PostgresAuditRepository(session)
        stored = repo.get(row.audit_id)

    assert stored == row


def test_raw_bars_upsert_is_idempotent() -> None:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    event_time = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [
        {
            "symbol": "BTCUSDT",
            "timeframe": "1m",
            "event_time": event_time,
            "fetch_time": event_time,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
            "source": "integration",
            "metadata_json": "{}",
        },
        {
            "symbol": "BTCUSDT",
            "timeframe": "1m",
            "event_time": event_time,
            "fetch_time": event_time,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
            "source": "integration",
            "metadata_json": "{}",
        },
    ]

    with session_scope(session_factory) as session:
        inserted = upsert_raw_bars(session, rows)

    with session_scope(session_factory) as session:
        count = session.execute(
            text("""
                select count(*)
                from raw_bars
                where symbol = 'BTCUSDT'
                  and timeframe = '1m'
                  and event_time = :event_time
                  and source = 'integration'
                """),
            {"event_time": event_time},
        ).scalar_one()

    assert inserted == 1
    assert count == 1


def test_strategy_comparison_snapshot_upsert_and_latest_fetch() -> None:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    ts_1 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    ts_2 = datetime(2026, 1, 1, 0, 5, tzinfo=UTC)

    with session_scope(session_factory) as session:
        repo = PostgresModelRepository(session)
        repo.upsert_strategy_comparison_snapshot(
            snapshot_id="scs_test_1",
            model_run_id="model_snapshot_1",
            symbol="BTCUSDT",
            timeframe="1m",
            feature_set_version="v1",
            created_at=ts_1,
            comparison_json={
                "model_run_id": "model_snapshot_1",
                "symbol": "BTCUSDT",
                "feature_set_version": "v1",
                "compared_strategies": ["ml_advisor", "mean_reversion"],
                "comparison_rows": [],
                "summary": {"best_strategy_by_total_return": "ml_advisor"},
                "quality_gates": {"ml_has_trades": True},
                "quality_gate_passed": True,
                "quality_gate_warnings": [],
                "comparison_config": {"n_folds": 3},
            },
        )
        repo.upsert_strategy_comparison_snapshot(
            snapshot_id="scs_test_2",
            model_run_id="model_snapshot_1",
            symbol="BTCUSDT",
            timeframe="1m",
            feature_set_version="v1",
            created_at=ts_2,
            comparison_json={
                "model_run_id": "model_snapshot_1",
                "symbol": "BTCUSDT",
                "feature_set_version": "v1",
                "compared_strategies": ["ml_advisor", "mean_reversion"],
                "comparison_rows": [],
                "summary": {"best_strategy_by_total_return": "mean_reversion"},
                "quality_gates": {"ml_has_trades": False},
                "quality_gate_passed": False,
                "quality_gate_warnings": ["ML advisor produced no trades."],
                "comparison_config": {"n_folds": 3},
            },
        )

    with session_scope(session_factory) as session:
        latest = PostgresModelRepository(session).latest_strategy_comparison_snapshot(
            model_run_id="model_snapshot_1"
        )

    assert latest is not None
    assert latest["snapshot_id"] == "scs_test_2"
    assert latest["model_run_id"] == "model_snapshot_1"
    assert latest["symbol"] == "BTCUSDT"
    assert latest["timeframe"] == "1m"
    assert latest["feature_set_version"] == "v1"
    assert latest["comparison_json"]["quality_gate_passed"] is False
