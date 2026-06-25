from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast


def read_json(path: str | Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(Path(path).read_text(encoding="utf-8")))
