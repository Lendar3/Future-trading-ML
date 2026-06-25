from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.core.enums import RiskStatus, SignalSide


@dataclass(frozen=True)
class RawBar:
    symbol: str
    timeframe: str
    event_time: datetime
    fetch_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str = "bybit"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Signal:
    signal_id: str
    symbol: str
    event_time: datetime
    strategy_name: str
    side: SignalSide
    score: float
    confidence: float
    entry_ref: float
    stop_ref: float
    take_ref: float
    ttl_sec: int
    rationale_json: dict[str, Any]
    feature_snapshot_hash: str
    risk_status: RiskStatus = RiskStatus.ACCEPTED
    risk_rejection_reason: str = ""


@dataclass(frozen=True)
class AuditLog:
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
