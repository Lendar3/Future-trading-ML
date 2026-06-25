from __future__ import annotations

import json
import os
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DEFAULT_API_BASE_URL = "http://localhost:8000"


class DashboardApiError(RuntimeError):
    pass


def api_base_url() -> str:
    return os.getenv("DASHBOARD_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


def get_json(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    query = f"?{urlencode(params)}" if params else ""
    url = f"{api_base_url()}{path}{query}"
    try:
        with urlopen(url, timeout=5) as response:
            return cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
    except HTTPError as exc:
        raise DashboardApiError(f"API returned {exc.code} for {path}") from exc
    except URLError as exc:
        raise DashboardApiError(f"API is unavailable at {api_base_url()}") from exc


def post_json(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    query = f"?{urlencode(params)}" if params else ""
    url = f"{api_base_url()}{path}{query}"
    request = Request(url=url, method="POST")
    try:
        with urlopen(request, timeout=30) as response:
            return cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
    except HTTPError as exc:
        raise DashboardApiError(f"API returned {exc.code} for {path}") from exc
    except URLError as exc:
        raise DashboardApiError(f"API is unavailable at {api_base_url()}") from exc


def get_paper_lifecycle_events(
    *,
    limit: int = 100,
    event_type: str | None = None,
    entity_type: str | None = None,
    entity_id: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"limit": limit}
    if event_type:
        params["event_type"] = event_type
    if entity_type:
        params["entity_type"] = entity_type
    if entity_id:
        params["entity_id"] = entity_id
    return get_json("/paper/lifecycle-events", params=params)


def get_paper_portfolio_audit(*, limit: int = 500) -> dict[str, Any]:
    return get_json("/paper/portfolio-audit", params={"limit": limit})
