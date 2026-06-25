from __future__ import annotations

from pathlib import Path


def load_init_sql() -> str:
    return Path("db/init.sql").read_text(encoding="utf-8")
