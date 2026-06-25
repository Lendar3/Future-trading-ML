from __future__ import annotations

from fastapi import APIRouter, Query

from app.core.enums import SystemMode
from app.core.modes import ModeState, transition_mode
from app.pipelines.demo_seed import seed_demo_data
from app.storage.database import build_engine, build_session_factory, session_scope
from app.storage.repositories import PostgresPaperRepository

router = APIRouter()
_state = ModeState()


@router.get("/mode/current")
def mode_current() -> dict[str, str]:
    return {"system_mode": _state.system_mode.value, "real_trading_enabled": "false"}


@router.post("/mode/set")
def mode_set(mode: SystemMode) -> dict[str, str]:
    global _state
    _state = transition_mode(_state, mode)
    _state.assert_no_live_trading()
    return {"system_mode": _state.system_mode.value, "real_trading_enabled": "false"}


@router.post("/demo/start")
def demo_start() -> dict[str, str]:
    return {"status": "started", "execution": "demo_simulation"}


@router.post("/demo/stop")
def demo_stop() -> dict[str, str]:
    return {"status": "stopped"}


@router.post("/demo/reset")
def demo_reset() -> dict[str, str]:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        snapshot_id = PostgresPaperRepository(session).reset_demo_account()
    return {"status": "reset", "snapshot_id": snapshot_id}


@router.post("/demo/seed")
def demo_seed(
    symbols: str = Query("BTCUSDT,ETHUSDT,SOLUSDT"),
    timeframe: str = Query("1m"),
    rows: int = Query(160, ge=80, le=5_000),
    source: str = Query("demo-seed"),
) -> dict[str, object]:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    selected_symbols = [symbol.strip() for symbol in symbols.split(",") if symbol.strip()]
    result = seed_demo_data(
        session_factory=session_factory,
        symbols=selected_symbols,
        timeframe=timeframe,
        rows=rows,
        source=source,
    )
    return {"status": "ok", **result.to_dict()}
