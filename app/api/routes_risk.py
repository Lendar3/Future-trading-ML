from __future__ import annotations

from fastapi import APIRouter

from app.core.risk_config import get_risk_config
from app.storage.database import build_engine, build_session_factory, session_scope
from app.storage.repositories import PostgresPaperRepository

router = APIRouter()


@router.get("/metrics/live")
def live_metrics() -> dict[str, object]:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        portfolio = PostgresPaperRepository(session).latest_portfolio()
    return {
        "status": "ok",
        "live_trading": "disabled",
        "portfolio": portfolio,
        "risk_limits": get_risk_config().to_dict(),
    }


@router.post("/risk/kill-switch")
def risk_kill_switch() -> dict[str, str]:
    return {"status": "activated"}
