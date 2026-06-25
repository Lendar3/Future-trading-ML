from __future__ import annotations

from datetime import UTC, datetime

from app.core.enums import RiskStatus, SignalSide
from app.core.types import Signal
from app.paper.risk import RiskLimits, RiskState, check_signal_risk


def _signal(
    *,
    side: SignalSide = SignalSide.LONG,
    entry_ref: float = 100.0,
    stop_ref: float = 90.0,
    take_ref: float = 110.0,
) -> Signal:
    return Signal(
        signal_id="sig-1",
        symbol="BTCUSDT",
        event_time=datetime(2026, 1, 1, tzinfo=UTC),
        strategy_name="test",
        side=side,
        score=1.0,
        confidence=0.8,
        entry_ref=entry_ref,
        stop_ref=stop_ref,
        take_ref=take_ref,
        ttl_sec=60,
        rationale_json={},
        feature_snapshot_hash="hash",
    )


def test_risk_rejects_missing_stop() -> None:
    status, reason = check_signal_risk(_signal(stop_ref=0), RiskState(equity=10_000), RiskLimits())
    assert status == RiskStatus.REJECTED
    assert reason == "missing stop reference"


def test_risk_rejects_missing_take() -> None:
    status, reason = check_signal_risk(_signal(take_ref=0), RiskState(equity=10_000), RiskLimits())
    assert status == RiskStatus.REJECTED
    assert reason == "missing take reference"


def test_risk_rejects_invalid_long_geometry() -> None:
    status, reason = check_signal_risk(
        _signal(side=SignalSide.LONG, entry_ref=100.0, stop_ref=101.0, take_ref=110.0),
        RiskState(equity=10_000),
        RiskLimits(),
    )
    assert status == RiskStatus.REJECTED
    assert reason == "invalid long stop/take geometry"


def test_risk_rejects_invalid_short_geometry() -> None:
    status, reason = check_signal_risk(
        _signal(side=SignalSide.SHORT, entry_ref=100.0, stop_ref=95.0, take_ref=105.0),
        RiskState(equity=10_000),
        RiskLimits(),
    )
    assert status == RiskStatus.REJECTED
    assert reason == "invalid short stop/take geometry"


def test_risk_rejects_kill_switch() -> None:
    status, reason = check_signal_risk(
        _signal(), RiskState(equity=10_000, kill_switch_active=True), RiskLimits()
    )
    assert status == RiskStatus.REJECTED
    assert reason == "kill switch active"


def test_risk_accepts_valid_signal() -> None:
    status, reason = check_signal_risk(_signal(), RiskState(equity=10_000), RiskLimits())
    assert status == RiskStatus.ACCEPTED
    assert reason == ""


def test_risk_accepts_valid_short_geometry() -> None:
    status, reason = check_signal_risk(
        _signal(side=SignalSide.SHORT, entry_ref=100.0, stop_ref=105.0, take_ref=95.0),
        RiskState(equity=10_000),
        RiskLimits(),
    )
    assert status == RiskStatus.ACCEPTED
    assert reason == ""
