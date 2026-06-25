from __future__ import annotations

from fastapi import APIRouter

from app.core.settings import get_settings

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    return {"status": "ok", "mode": settings.system_mode.value, "live_trading": "disabled"}
