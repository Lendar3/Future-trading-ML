from __future__ import annotations

from fastapi import FastAPI

from app.api import (
    routes_backtests,
    routes_data,
    routes_health,
    routes_models,
    routes_modes,
    routes_paper,
    routes_risk,
    routes_signals,
    routes_ui,
)

app = FastAPI(
    title="Bybit Futures Edge Lab",
    version="0.1.0",
    description="Research, backtest, paper/demo, and real-advisory API. Live trading is disabled.",
)

for router in (
    routes_health.router,
    routes_data.router,
    routes_models.router,
    routes_backtests.router,
    routes_signals.router,
    routes_paper.router,
    routes_ui.router,
    routes_risk.router,
    routes_modes.router,
):
    app.include_router(router)
