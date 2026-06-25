from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Body, HTTPException, Query

from app.paper.approval import build_paper_demo_approval_report
from app.paper.exit_policy import simulate_paper_exit_from_bars
from app.paper.intents import build_paper_intent_candidate
from app.paper.service import (
    PaperServiceError,
    close_paper_fill_service,
    persist_paper_intent_service,
    simulate_order_from_intent_service,
)
from app.reports.paper_portfolio_audit import build_paper_portfolio_audit_report
from app.reports.paper_simulation import build_paper_simulation_audit_report
from app.reports.paper_track_record import build_paper_track_record_report
from app.reports.readiness_loader import load_readiness_report_payload
from app.storage.database import build_engine, build_session_factory, session_scope
from app.storage.repositories import PostgresPaperRepository

router = APIRouter()


def _parse_iso_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    raise ValueError("Expected datetime or ISO datetime string.")


@router.get("/paper/portfolio")
def paper_portfolio() -> dict[str, object]:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        return PostgresPaperRepository(session).latest_portfolio()


@router.get("/paper/orders")
def paper_orders(limit: int = Query(50, ge=1, le=500)) -> dict[str, list[dict[str, object]]]:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        return {"orders": PostgresPaperRepository(session).latest_orders(limit=limit)}


@router.get("/paper/fills")
def paper_fills(limit: int = Query(50, ge=1, le=500)) -> dict[str, list[dict[str, object]]]:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        return {"fills": PostgresPaperRepository(session).latest_fills(limit=limit)}


@router.get("/paper/approval/{model_run_id}")
def paper_demo_approval(model_run_id: str) -> dict[str, object]:
    try:
        readiness = load_readiness_report_payload(model_run_id=model_run_id)
    except FileNotFoundError:
        readiness = None

    return build_paper_demo_approval_report(
        model_run_id=model_run_id,
        readiness_report=readiness,
    )


@router.post("/paper/intents/from-signal")
def paper_intent_from_signal(
    payload: Annotated[dict[str, Any], Body(...)],
) -> dict[str, object]:
    model_run_id = str(payload.get("model_run_id", "")).strip()
    if not model_run_id:
        raise HTTPException(status_code=400, detail="model_run_id is required.")

    signal_payload = payload.get("signal_payload")
    resolved_signal_payload: dict[str, Any]
    if isinstance(signal_payload, dict):
        resolved_signal_payload = dict(signal_payload)
    else:
        resolved_signal_payload = {
            key: value
            for key, value in payload.items()
            if key not in {"model_run_id", "signal_payload"}
        }

    try:
        readiness = load_readiness_report_payload(model_run_id=model_run_id)
    except FileNotFoundError:
        readiness = None
    approval = build_paper_demo_approval_report(
        model_run_id=model_run_id,
        readiness_report=readiness,
    )
    intent = build_paper_intent_candidate(
        model_run_id=model_run_id,
        signal_payload=resolved_signal_payload,
        approval_report=approval,
    )

    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        repo = PostgresPaperRepository(session)
        try:
            return persist_paper_intent_service(repo=repo, intent_payload=intent)
        except PaperServiceError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@router.get("/paper/intents")
def paper_intents(
    limit: int = Query(50, ge=1, le=500),
    status: str | None = Query(None),
) -> dict[str, list[dict[str, object]]]:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        intents = PostgresPaperRepository(session).latest_paper_intents(
            limit=limit,
            status=status,
        )
    return {"intents": intents}


@router.get("/paper/simulation-audit")
def paper_simulation_audit(limit: int = Query(50, ge=1, le=500)) -> dict[str, object]:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        repo = PostgresPaperRepository(session)
        report = build_paper_simulation_audit_report(
            paper_intents=repo.latest_paper_intents(limit=limit),
            paper_orders=repo.latest_orders(limit=limit),
            paper_fills=repo.latest_fills(limit=limit),
            paper_intent_executions=repo.latest_paper_intent_executions(limit=limit),
            portfolio_snapshot=repo.latest_portfolio(),
            limit=limit,
        )
    return report


@router.get("/paper/track-record")
def paper_track_record(limit: int = Query(500, ge=1, le=5000)) -> dict[str, object]:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        repo = PostgresPaperRepository(session)
        report = build_paper_track_record_report(
            paper_orders=repo.latest_orders(limit=limit),
            paper_fills=repo.latest_fills(limit=limit),
            portfolio_snapshots=repo.latest_portfolio_snapshots(limit=limit),
            current_portfolio_snapshot=repo.latest_portfolio(),
            limit=limit,
        )
    return report


@router.get("/paper/portfolio-audit")
def paper_portfolio_audit(limit: int = Query(500, ge=1, le=5000)) -> dict[str, object]:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        repo = PostgresPaperRepository(session)
        report = build_paper_portfolio_audit_report(
            paper_fills=repo.latest_fills(limit=limit),
            paper_trade_closes=repo.latest_paper_trade_closes(limit=limit),
            portfolio_snapshots=repo.latest_portfolio_snapshots(limit=limit),
            current_portfolio_snapshot=repo.latest_portfolio(),
            limit=limit,
        )
    return report


@router.get("/paper/lifecycle-events")
def paper_lifecycle_events(
    limit: int = Query(100, ge=1, le=500),
    event_type: str | None = Query(None),
    entity_type: str | None = Query(None),
    entity_id: str | None = Query(None),
) -> dict[str, object]:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        repo = PostgresPaperRepository(session)
        events = repo.latest_paper_lifecycle_events(
            limit=limit,
            event_type=event_type,
            entity_type=entity_type,
            entity_id=entity_id,
        )
    return {"events": events, "count": len(events)}


@router.post("/paper/fills/{fill_id}/close")
def close_paper_fill_manually(
    fill_id: str,
    payload: Annotated[dict[str, Any] | None, Body()] = None,
) -> dict[str, object]:
    close_payload = payload if isinstance(payload, dict) else {}
    raw_exit_price = close_payload.get("exit_price")
    if raw_exit_price is None:
        raise HTTPException(status_code=400, detail="exit_price is required and must be numeric.")
    try:
        exit_price = float(raw_exit_price)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="exit_price is required and must be numeric.",
        ) from exc
    raw_exit_fee = close_payload.get("exit_fee", 0.0)
    try:
        exit_fee = float(raw_exit_fee)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="exit_fee must be numeric.") from exc
    close_reason = str(close_payload.get("close_reason", "manual_exit")).strip() or "manual_exit"
    raw_closed_at = close_payload.get("closed_at")
    closed_at = datetime.now(UTC) if raw_closed_at is None else _parse_iso_datetime(raw_closed_at)

    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        repo = PostgresPaperRepository(session)
        try:
            return close_paper_fill_service(
                repo=repo,
                fill_id=fill_id,
                exit_price=exit_price,
                exit_fee=exit_fee,
                close_reason=close_reason,
                closed_at=closed_at,
            )
        except PaperServiceError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@router.post("/paper/fills/{fill_id}/exit-preview")
def preview_paper_fill_exit(
    fill_id: str,
    payload: Annotated[dict[str, Any] | None, Body()] = None,
) -> dict[str, object]:
    preview_payload = payload if isinstance(payload, dict) else {}
    signal_payload = preview_payload.get("signal")
    ohlc_bars_payload = preview_payload.get("ohlc_bars")

    if not isinstance(signal_payload, dict):
        raise HTTPException(status_code=400, detail="signal is required and must be an object.")
    if not isinstance(ohlc_bars_payload, list):
        raise HTTPException(
            status_code=400,
            detail="ohlc_bars is required and must be a list of objects.",
        )
    if any(not isinstance(item, dict) for item in ohlc_bars_payload):
        raise HTTPException(status_code=400, detail="ohlc_bars must be a list of objects.")

    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        repo = PostgresPaperRepository(session)
        fill = repo.get_paper_fill(fill_id)
        if fill is None:
            raise HTTPException(status_code=404, detail="Paper fill not found.")
        try:
            exit_preview = simulate_paper_exit_from_bars(
                fill_payload=fill,
                signal_payload=dict(signal_payload),
                ohlc_bars=[dict(item) for item in ohlc_bars_payload],
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "fill_id": fill_id,
        "exit_preview": exit_preview,
        "preview_only": True,
        "simulation_only": True,
        "manual_trigger": True,
        "real_execution": False,
    }


@router.post("/paper/intents/{intent_id}/simulate-order")
def simulate_paper_order_from_intent(
    intent_id: str,
    payload: Annotated[dict[str, Any] | None, Body()] = None,
) -> dict[str, object]:
    execution_config = payload if isinstance(payload, dict) else {}
    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        repo = PostgresPaperRepository(session)
        try:
            return simulate_order_from_intent_service(
                repo=repo,
                intent_id=intent_id,
                execution_config=execution_config,
            )
        except PaperServiceError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
