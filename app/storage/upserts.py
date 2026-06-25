from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

BAR_TABLES = {"raw_bars", "raw_mark_bars", "raw_index_bars", "ext_market_bars"}


def deduplicate_by_key(
    rows: Iterable[dict[str, Any]], key_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    output: list[dict[str, Any]] = []
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def upsert_bar_rows(
    session: Session,
    rows: Iterable[dict[str, Any]],
    table_name: str = "raw_bars",
) -> int:
    if table_name not in BAR_TABLES:
        raise ValueError(f"Unsupported bar table: {table_name}")
    unique_rows = deduplicate_by_key(rows, ("symbol", "timeframe", "event_time", "source"))
    if not unique_rows:
        return 0
    session.execute(
        text(f"""
            insert into {table_name} (
                symbol,
                timeframe,
                event_time,
                fetch_time,
                open,
                high,
                low,
                close,
                volume,
                source,
                metadata_json
            )
            values (
                :symbol,
                :timeframe,
                :event_time,
                :fetch_time,
                :open,
                :high,
                :low,
                :close,
                :volume,
                :source,
                cast(:metadata_json as jsonb)
            )
            on conflict (symbol, timeframe, event_time, source) do update set
                fetch_time = excluded.fetch_time,
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume,
                metadata_json = excluded.metadata_json
            """),
        unique_rows,
    )
    return len(unique_rows)


def upsert_raw_bars(session: Session, rows: Iterable[dict[str, Any]]) -> int:
    return upsert_bar_rows(session=session, rows=rows, table_name="raw_bars")


def _ensure_public_market_indexes(session: Session) -> None:
    session.execute(
        text(
            "create unique index if not exists raw_funding_symbol_event_source_idx "
            "on raw_funding (symbol, event_time, source)"
        )
    )
    session.execute(
        text(
            "create unique index if not exists raw_open_interest_symbol_event_source_idx "
            "on raw_open_interest (symbol, event_time, source)"
        )
    )
    session.execute(
        text(
            "create unique index if not exists raw_long_short_ratio_symbol_event_source_idx "
            "on raw_long_short_ratio (symbol, event_time, source)"
        )
    )


def upsert_funding_rows(session: Session, rows: Iterable[dict[str, Any]]) -> int:
    _ensure_public_market_indexes(session)
    unique_rows = deduplicate_by_key(rows, ("symbol", "event_time", "source"))
    if not unique_rows:
        return 0
    session.execute(
        text("""
            insert into raw_funding (
                symbol,
                event_time,
                fetch_time,
                funding_rate,
                source,
                metadata_json
            )
            values (
                :symbol,
                :event_time,
                :fetch_time,
                :funding_rate,
                :source,
                cast(:metadata_json as jsonb)
            )
            on conflict (symbol, event_time, source) do update set
                fetch_time = excluded.fetch_time,
                funding_rate = excluded.funding_rate,
                metadata_json = excluded.metadata_json
            """),
        unique_rows,
    )
    return len(unique_rows)


def upsert_open_interest_rows(session: Session, rows: Iterable[dict[str, Any]]) -> int:
    _ensure_public_market_indexes(session)
    unique_rows = deduplicate_by_key(rows, ("symbol", "event_time", "source"))
    if not unique_rows:
        return 0
    session.execute(
        text("""
            insert into raw_open_interest (
                symbol,
                event_time,
                fetch_time,
                open_interest,
                source,
                metadata_json
            )
            values (
                :symbol,
                :event_time,
                :fetch_time,
                :open_interest,
                :source,
                cast(:metadata_json as jsonb)
            )
            on conflict (symbol, event_time, source) do update set
                fetch_time = excluded.fetch_time,
                open_interest = excluded.open_interest,
                metadata_json = excluded.metadata_json
            """),
        unique_rows,
    )
    return len(unique_rows)


def upsert_long_short_ratio_rows(session: Session, rows: Iterable[dict[str, Any]]) -> int:
    _ensure_public_market_indexes(session)
    unique_rows = deduplicate_by_key(rows, ("symbol", "event_time", "source"))
    if not unique_rows:
        return 0
    session.execute(
        text("""
            insert into raw_long_short_ratio (
                symbol,
                event_time,
                fetch_time,
                long_short_ratio,
                source,
                metadata_json
            )
            values (
                :symbol,
                :event_time,
                :fetch_time,
                :long_short_ratio,
                :source,
                cast(:metadata_json as jsonb)
            )
            on conflict (symbol, event_time, source) do update set
                fetch_time = excluded.fetch_time,
                long_short_ratio = excluded.long_short_ratio,
                metadata_json = excluded.metadata_json
            """),
        unique_rows,
    )
    return len(unique_rows)


def upsert_signals(session: Session, rows: Iterable[dict[str, Any]]) -> int:
    unique_rows = deduplicate_by_key(rows, ("signal_id",))
    if not unique_rows:
        return 0
    session.execute(
        text("""
            insert into signals (
                signal_id,
                symbol,
                event_time,
                strategy_name,
                side,
                score,
                confidence,
                rationale_json
            )
            values (
                :signal_id,
                :symbol,
                :event_time,
                :strategy_name,
                :side,
                :score,
                :confidence,
                cast(:rationale_json as jsonb)
            )
            on conflict (signal_id) do update set
                symbol = excluded.symbol,
                event_time = excluded.event_time,
                strategy_name = excluded.strategy_name,
                side = excluded.side,
                score = excluded.score,
                confidence = excluded.confidence,
                rationale_json = excluded.rationale_json
            """),
        unique_rows,
    )
    return len(unique_rows)
