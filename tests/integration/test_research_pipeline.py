from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from app.pipelines.research_pipeline import run_research_pipeline_from_postgres
from app.storage.database import build_engine, build_session_factory, session_scope
from app.storage.upserts import upsert_raw_bars


def _synthetic_bar_rows() -> list[dict]:
    start = datetime(2026, 1, 2, tzinfo=UTC)
    rows = []
    for index in range(80):
        event_time = start + timedelta(minutes=index)
        base = 100 + math.sin(index / 4) * 2
        close = base - 8 if index >= 76 else base
        rows.append(
            {
                "symbol": "BTCUSDT",
                "timeframe": "1m",
                "event_time": event_time,
                "fetch_time": event_time,
                "open": close - 0.2,
                "high": close + 0.8,
                "low": close - 0.8,
                "close": close,
                "volume": 10 + index,
                "source": "pipeline-test",
                "metadata_json": "{}",
            }
        )
    return rows


def test_research_pipeline_writes_signal_and_audit_log() -> None:
    engine = build_engine()
    session_factory = build_session_factory(engine)

    with session_scope(session_factory) as session:
        inserted = upsert_raw_bars(session, _synthetic_bar_rows())

    result = run_research_pipeline_from_postgres(
        session_factory=session_factory,
        symbol="BTCUSDT",
        timeframe="1m",
        source="pipeline-test",
    )

    with session_scope(session_factory) as session:
        signal_count = session.execute(
            text("select count(*) from signals where signal_id = :signal_id"),
            {"signal_id": result.signal_id},
        ).scalar_one()
        audit_count = session.execute(
            text("select count(*) from signal_audit_logs where audit_id = :audit_id"),
            {"audit_id": result.audit_id},
        ).scalar_one()
        snapshot = session.execute(
            text("""
                select feature_snapshot_json
                from signal_audit_logs
                where audit_id = :audit_id
                """),
            {"audit_id": result.audit_id},
        ).scalar_one()

    assert inserted == 80
    assert result.dataset_rows == 80
    assert result.decision in {"long", "short", "no_trade"}
    assert signal_count == 1
    assert audit_count == 1
    assert "feature_zscore_close_20" in snapshot
