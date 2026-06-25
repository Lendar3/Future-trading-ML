from __future__ import annotations

from datetime import UTC, datetime

from app.storage.models import SignalAuditRow
from app.storage.repositories import InMemoryAuditRepository


def test_audit_log_creation() -> None:
    row = SignalAuditRow(
        audit_id="audit-1",
        signal_id="sig-1",
        symbol="BTCUSDT",
        event_time=datetime(2026, 1, 1, tzinfo=UTC),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        strategy_name="mean_reversion",
        model_run_id=None,
        feature_set_version="v1",
        feature_snapshot_json={"feature_ret_1m": 0.01},
        rationale_json={"reason": "test"},
        risk_checks_json={"status": "accepted"},
        cost_assumptions_json={"taker_fee_bp": 5.5},
        decision="no_trade",
        decision_reason="test",
        system_mode="research",
        code_version="local",
        config_hash="hash",
    )
    repo = InMemoryAuditRepository()
    repo.add(row)
    assert repo.latest() == row
