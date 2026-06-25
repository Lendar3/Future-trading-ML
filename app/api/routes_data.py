from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.ingestion.bybit_rest import BybitPublicRestClient, BybitPublicRestError
from app.ingestion.public_backfill import (
    default_backfill_window,
    parse_cli_datetime,
    run_public_backfill,
)
from app.pipelines.research_pipeline import run_research_pipeline_from_postgres
from app.storage.database import build_engine, build_session_factory

router = APIRouter()


@router.get("/symbols")
def symbols() -> dict[str, list[str]]:
    return {"symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"]}


@router.post("/collect/backfill")
def collect_backfill() -> dict[str, str]:
    return {"status": "queued", "note": "provider wiring pending credentials"}


@router.post("/collect/public-backfill")
def collect_public_backfill(
    symbol: str = Query("BTCUSDT"),
    timeframe: str = Query("1m"),
    start: str | None = Query(None),
    end: str | None = Query(None),
    hours: int = Query(24, ge=1, le=24 * 30),
) -> dict[str, object]:
    if start and end:
        start_dt = parse_cli_datetime(start)
        end_dt = parse_cli_datetime(end)
    else:
        start_dt, end_dt = default_backfill_window(hours=hours)

    engine = build_engine()
    session_factory = build_session_factory(engine)
    try:
        result = run_public_backfill(
            session_factory=session_factory,
            client=BybitPublicRestClient(),
            symbol=symbol,
            timeframe=timeframe,
            start=start_dt,
            end=end_dt,
        )
    except (BybitPublicRestError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"status": "ok", **result.to_dict()}


@router.post("/features/build")
def features_build(
    symbol: str = Query("BTCUSDT"),
    timeframe: str = Query("1m"),
    source: str | None = Query(None),
    run_research_pipeline: bool = Query(False),
) -> dict[str, object]:
    if not run_research_pipeline:
        return {
            "status": "ok",
            "feature_set_version": "v1",
            "mode": "contract_only",
        }

    engine = build_engine()
    session_factory = build_session_factory(engine)
    try:
        result = run_research_pipeline_from_postgres(
            session_factory=session_factory,
            symbol=symbol,
            timeframe=timeframe,
            source=source,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "status": "ok",
        "feature_set_version": "v1",
        "dataset_rows": result.dataset_rows,
        "signal_id": result.signal_id,
        "audit_id": result.audit_id,
        "decision": result.decision,
    }
