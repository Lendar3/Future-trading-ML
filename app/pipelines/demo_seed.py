from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import sessionmaker

from app.pipelines.research_pipeline import (
    ResearchPipelineResult,
    run_research_pipeline_from_postgres,
)
from app.storage.database import build_engine, build_session_factory, session_scope
from app.storage.upserts import upsert_raw_bars


@dataclass(frozen=True)
class DemoSeedResult:
    symbols: list[str]
    timeframe: str
    source: str
    bars_upserted: int
    pipeline_results: list[ResearchPipelineResult]

    def to_dict(self) -> dict[str, object]:
        return {
            "symbols": self.symbols,
            "timeframe": self.timeframe,
            "source": self.source,
            "bars_upserted": self.bars_upserted,
            "pipeline_results": [asdict(result) for result in self.pipeline_results],
        }


def generate_demo_bar_rows(
    symbol: str,
    timeframe: str = "1m",
    rows: int = 160,
    source: str = "demo-seed",
) -> list[dict[str, object]]:
    start = datetime(2026, 1, 4, tzinfo=UTC)
    symbol_offset = {"BTCUSDT": 100.0, "ETHUSDT": 60.0, "SOLUSDT": 25.0}.get(symbol, 50.0)
    output: list[dict[str, object]] = []
    for index in range(rows):
        event_time = start + timedelta(minutes=index)
        wave = math.sin(index / 8) * 2.5
        trend = index * 0.03
        shock = -7.5 if index >= rows - 6 and symbol == "BTCUSDT" else 0.0
        close = symbol_offset + wave + trend + shock
        output.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "event_time": event_time,
                "fetch_time": event_time,
                "open": close - 0.15,
                "high": close + 0.75,
                "low": close - 0.75,
                "close": close,
                "volume": 100 + index * 2,
                "source": source,
                "metadata_json": '{"seed": "demo"}',
            }
        )
    return output


def seed_demo_data(
    session_factory: sessionmaker,
    symbols: list[str] | None = None,
    timeframe: str = "1m",
    rows: int = 160,
    source: str = "demo-seed",
) -> DemoSeedResult:
    selected_symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    all_rows: list[dict[str, object]] = []
    for symbol in selected_symbols:
        all_rows.extend(
            generate_demo_bar_rows(
                symbol=symbol,
                timeframe=timeframe,
                rows=rows,
                source=source,
            )
        )

    with session_scope(session_factory) as session:
        bars_upserted = upsert_raw_bars(session, all_rows)

    pipeline_results = [
        run_research_pipeline_from_postgres(
            session_factory=session_factory,
            symbol=symbol,
            timeframe=timeframe,
            source=source,
        )
        for symbol in selected_symbols
    ]
    return DemoSeedResult(
        symbols=selected_symbols,
        timeframe=timeframe,
        source=source,
        bars_upserted=bars_upserted,
        pipeline_results=pipeline_results,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed local demo data and run research pipeline.")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--rows", type=int, default=160)
    parser.add_argument("--source", default="demo-seed")
    args = parser.parse_args()

    engine = build_engine()
    session_factory = build_session_factory(engine)
    result = seed_demo_data(
        session_factory=session_factory,
        symbols=[symbol.strip() for symbol in args.symbols.split(",") if symbol.strip()],
        timeframe=args.timeframe,
        rows=args.rows,
        source=args.source,
    )
    print(result.to_dict())


if __name__ == "__main__":
    main()
