from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import sessionmaker

from app.core.timeframes import app_timeframe_to_bybit_interval, normalize_app_timeframe
from app.core.types import RawBar
from app.ingestion.bybit_rest import BybitPublicRestClient, raw_bar_to_upsert_row
from app.storage.database import build_engine, build_session_factory, session_scope
from app.storage.upserts import (
    upsert_bar_rows,
    upsert_funding_rows,
    upsert_long_short_ratio_rows,
    upsert_open_interest_rows,
    upsert_raw_bars,
)


@dataclass(frozen=True)
class PublicBackfillResult:
    symbol: str
    timeframe: str
    bybit_interval: str
    start: str
    end: str
    regular_bars: int
    mark_bars: int
    index_bars: int
    funding_points: int
    open_interest_points: int
    long_short_ratio_points: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def parse_cli_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def default_backfill_window(hours: int = 24) -> tuple[datetime, datetime]:
    end = datetime.now(tz=UTC)
    return end - timedelta(hours=hours), end


def _as_app_timeframe_bars(rows: list[RawBar], app_timeframe: str) -> list[RawBar]:
    return [replace(row, timeframe=app_timeframe) for row in rows]


def run_public_backfill(
    session_factory: sessionmaker,
    client: BybitPublicRestClient,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    include_mark: bool = True,
    include_index: bool = True,
    include_funding: bool = True,
    include_open_interest: bool = True,
    include_long_short: bool = True,
    open_interest_interval: str = "5min",
    long_short_period: str = "5min",
) -> PublicBackfillResult:
    app_timeframe = normalize_app_timeframe(timeframe)
    bybit_interval = app_timeframe_to_bybit_interval(app_timeframe)

    regular_bars = client.fetch_klines(
        symbol=symbol,
        timeframe=bybit_interval,
        start=start,
        end=end,
    )
    mark_bars = (
        client.fetch_mark_price_klines(
            symbol=symbol,
            timeframe=bybit_interval,
            start=start,
            end=end,
        )
        if include_mark
        else []
    )
    index_bars = (
        client.fetch_index_price_klines(
            symbol=symbol,
            timeframe=bybit_interval,
            start=start,
            end=end,
        )
        if include_index
        else []
    )
    funding = (
        client.fetch_funding_history(symbol=symbol, start=start, end=end) if include_funding else []
    )
    open_interest = (
        client.fetch_open_interest(
            symbol=symbol,
            interval_time=open_interest_interval,
            start=start,
            end=end,
        )
        if include_open_interest
        else []
    )
    long_short = (
        client.fetch_long_short_ratio(
            symbol=symbol,
            period=long_short_period,
            start=start,
            end=end,
        )
        if include_long_short
        else []
    )

    regular_rows = _as_app_timeframe_bars(regular_bars, app_timeframe)
    mark_rows = _as_app_timeframe_bars(mark_bars, app_timeframe)
    index_rows = _as_app_timeframe_bars(index_bars, app_timeframe)

    with session_scope(session_factory) as session:
        regular_count = upsert_raw_bars(
            session,
            [raw_bar_to_upsert_row(row) for row in regular_rows],
        )
        mark_count = upsert_bar_rows(
            session,
            [raw_bar_to_upsert_row(row) for row in mark_rows],
            table_name="raw_mark_bars",
        )
        index_count = upsert_bar_rows(
            session,
            [raw_bar_to_upsert_row(row) for row in index_rows],
            table_name="raw_index_bars",
        )
        funding_count = upsert_funding_rows(session, funding)
        open_interest_count = upsert_open_interest_rows(session, open_interest)
        long_short_count = upsert_long_short_ratio_rows(session, long_short)

    return PublicBackfillResult(
        symbol=symbol,
        timeframe=app_timeframe,
        bybit_interval=bybit_interval,
        start=start.isoformat(),
        end=end.isoformat(),
        regular_bars=regular_count,
        mark_bars=mark_count,
        index_bars=index_count,
        funding_points=funding_count,
        open_interest_points=open_interest_count,
        long_short_ratio_points=long_short_count,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Backfill public Bybit V5 market data.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--hours", type=int, default=24)
    args = parser.parse_args(argv)

    if args.start and args.end:
        start = parse_cli_datetime(args.start)
        end = parse_cli_datetime(args.end)
    else:
        start, end = default_backfill_window(hours=args.hours)

    engine = build_engine()
    session_factory = build_session_factory(engine)
    result = run_public_backfill(
        session_factory=session_factory,
        client=BybitPublicRestClient(),
        symbol=args.symbol,
        timeframe=args.timeframe,
        start=start,
        end=end,
    )
    print(result.to_dict())


if __name__ == "__main__":
    main()
