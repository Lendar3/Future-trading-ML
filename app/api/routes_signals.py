from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.storage.database import build_engine, build_session_factory, session_scope
from app.storage.repositories import PostgresAuditRepository, PostgresSignalRepository

router = APIRouter()


@router.get("/signals/latest")
def latest_signals(
    symbol: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
) -> dict[str, list[dict[str, object]]]:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        repo = PostgresSignalRepository(session)
        return {"signals": repo.latest(symbol=symbol, limit=limit)}


@router.get("/signals/history")
def signal_history() -> dict[str, list[dict[str, str]]]:
    return {"signals": []}


@router.get("/signals/{signal_id}/explain")
def explain_signal(signal_id: str) -> dict[str, object]:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        repo = PostgresAuditRepository(session)
        audit = repo.get_by_signal_id(signal_id)

    if audit is None:
        raise HTTPException(status_code=404, detail=f"No audit log found for signal {signal_id}.")

    return {
        "audit_id": audit.audit_id,
        "signal_id": audit.signal_id,
        "symbol": audit.symbol,
        "event_time": audit.event_time.isoformat(),
        "created_at": audit.created_at.isoformat(),
        "strategy_name": audit.strategy_name,
        "model_run_id": audit.model_run_id,
        "feature_set_version": audit.feature_set_version,
        "feature_snapshot": audit.feature_snapshot_json,
        "rationale": audit.rationale_json,
        "risk_checks": audit.risk_checks_json,
        "cost_assumptions": audit.cost_assumptions_json,
        "decision": audit.decision,
        "decision_reason": audit.decision_reason,
        "system_mode": audit.system_mode,
        "code_version": audit.code_version,
        "config_hash": audit.config_hash,
    }
