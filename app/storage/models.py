from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class SignalAuditRow:
    audit_id: str
    signal_id: str
    symbol: str
    event_time: datetime
    created_at: datetime
    strategy_name: str
    model_run_id: str | None
    feature_set_version: str
    feature_snapshot_json: dict[str, Any]
    rationale_json: dict[str, Any]
    risk_checks_json: dict[str, Any]
    cost_assumptions_json: dict[str, Any]
    decision: str
    decision_reason: str
    system_mode: str
    code_version: str
    config_hash: str
