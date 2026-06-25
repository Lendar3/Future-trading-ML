from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text

from app.ingestion.bybit_rest import BybitPublicRestClient
from app.ingestion.public_backfill import run_public_backfill
from app.storage.database import build_engine, build_session_factory, session_scope


class FixtureBybitClient(BybitPublicRestClient):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "seen_requests", [])

    def _fixture(self, name: str) -> dict[str, Any]:
        return json.loads(Path(f"tests/fixtures/bybit/{name}.json").read_text(encoding="utf-8"))

    def _request(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        self.seen_requests.append({"path": path, "params": dict(params)})
        if path == "/v5/market/kline":
            return self._fixture("kline")
        if path == "/v5/market/mark-price-kline":
            return self._fixture("mark_price_kline")
        if path == "/v5/market/index-price-kline":
            return self._fixture("mark_price_kline")
        if path == "/v5/market/funding/history":
            return self._fixture("funding")
        if path == "/v5/market/open-interest":
            return self._fixture("open_interest")
        if path == "/v5/market/account-ratio":
            return self._fixture("long_short_ratio")
        raise AssertionError(path)


def test_public_backfill_with_fixture_responses_persists_market_data() -> None:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    source = "bybit_fixture_timeframe_hotfix"
    client = FixtureBybitClient(source=source)
    result = run_public_backfill(
        session_factory=session_factory,
        client=client,
        symbol="BTCUSDT",
        timeframe="1m",
        start=datetime(2026, 5, 8, tzinfo=UTC),
        end=datetime(2026, 5, 8, 0, 5, tzinfo=UTC),
    )

    assert result.timeframe == "1m"
    assert result.bybit_interval == "1"
    assert result.regular_bars == 2
    assert result.mark_bars == 1
    assert result.index_bars == 1
    assert result.funding_points == 1
    assert result.open_interest_points == 1
    assert result.long_short_ratio_points == 1

    with session_scope(session_factory) as session:
        raw_count = session.execute(
            text("""
            select count(*)
            from raw_bars
            where source = :source
              and timeframe = '1m'
            """),
            {"source": source},
        ).scalar_one()
        legacy_timeframe_count = session.execute(
            text("""
            select count(*)
            from raw_bars
            where source = :source
              and timeframe = '1'
            """),
            {"source": source},
        ).scalar_one()
        funding_count = session.execute(
            text("select count(*) from raw_funding where source = :source"),
            {"source": source},
        ).scalar_one()

    assert raw_count >= 2
    assert legacy_timeframe_count == 0
    assert funding_count >= 1
    kline_request = next(
        item for item in client.seen_requests if item["path"] == "/v5/market/kline"
    )
    assert kline_request["params"]["interval"] == "1"
