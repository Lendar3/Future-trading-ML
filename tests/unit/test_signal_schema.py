from __future__ import annotations

from datetime import UTC, datetime

from app.core.enums import SignalSide
from app.core.types import Signal


def test_signal_schema_contains_required_fields() -> None:
    signal = Signal(
        signal_id="sig-1",
        symbol="BTCUSDT",
        event_time=datetime(2026, 1, 1, tzinfo=UTC),
        strategy_name="mean_reversion",
        side=SignalSide.NO_TRADE,
        score=0.0,
        confidence=0.0,
        entry_ref=0.0,
        stop_ref=0.0,
        take_ref=0.0,
        ttl_sec=60,
        rationale_json={"reason": "test"},
        feature_snapshot_hash="hash",
    )
    assert signal.symbol == "BTCUSDT"
    assert signal.side == SignalSide.NO_TRADE
