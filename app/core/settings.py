from __future__ import annotations

import os
from dataclasses import dataclass

from app.core.enums import SystemMode
from app.core.safety import assert_real_trading_disabled


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_env: str = os.getenv("APP_ENV", "local")
    database_url: str = os.getenv(
        "DATABASE_URL", "postgresql+psycopg://bybit:change_me@localhost:5432/bybit_edge"
    )
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    system_mode: SystemMode = SystemMode(os.getenv("SYSTEM_MODE", SystemMode.RESEARCH.value))
    real_trading_enabled: bool = _env_bool("REAL_TRADING_ENABLED", False)
    demo_trading_enabled: bool = _env_bool("DEMO_TRADING_ENABLED", False)
    paper_trading_enabled: bool = _env_bool("PAPER_TRADING_ENABLED", False)

    def validate_safety(self) -> None:
        assert_real_trading_disabled(self.real_trading_enabled)


def get_settings() -> Settings:
    settings = Settings()
    settings.validate_safety()
    return settings
