"""Research pipeline: bars → features → labels → signals → audit log.

Feature-set modes
-----------------
* ``feature_set_name="v1"`` (default) — legacy OHLCV path.
* ``feature_set_name="v1.6_public"`` — enriched path that also fetches and
  joins mark/index/funding/OI/long-short tables from PostgreSQL.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

import pandas as pd
from sqlalchemy.orm import sessionmaker

from app.backtest.costs import CostAssumptions
from app.core.clock import utc_now
from app.features.dataset_builder import (
    FEATURE_SET_PUBLIC,
    build_dataset_from_bars,
    build_dataset_from_public_tables,
)
from app.storage.database import session_scope
from app.storage.models import SignalAuditRow
from app.storage.repositories import (
    PostgresAuditRepository,
    PostgresMarketDataRepository,
    PostgresPublicMarketRepository,
)
from app.storage.upserts import upsert_signals
from app.strategies.mean_reversion import generate_mean_reversion_signals


@dataclass(frozen=True)
class ResearchPipelineResult:
    dataset_rows: int
    signal_id: str
    audit_id: str
    decision: str


def _json_ready(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _feature_snapshot(feature_row: pd.Series) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for raw_key, value in feature_row.items():
        key = str(raw_key)
        if key.startswith("feature_") or key in {"symbol", "timeframe", "event_time"}:
            snapshot[key] = _json_ready(value)
    return snapshot


def _signal_rows_for_storage(signals: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in signals.itertuples(index=False):
        row_any = cast(Any, row)
        rows.append(
            {
                "signal_id": row_any.signal_id,
                "symbol": row_any.symbol,
                "event_time": row_any.event_time,
                "strategy_name": row_any.strategy_name,
                "side": row_any.side,
                "score": float(row_any.score),
                "confidence": float(row_any.confidence),
                "rationale_json": json.dumps(row_any.rationale_json),
            }
        )
    return rows


def run_research_pipeline_from_postgres(
    session_factory: sessionmaker,
    symbol: str,
    timeframe: str,
    source: str | None = None,
    feature_set_name: str = "v1",
) -> ResearchPipelineResult:
    """Run the research pipeline.

    When ``feature_set_name == "v1.6_public"`` the pipeline fetches all
    five public market tables and builds the enriched dataset.  For any
    other value the legacy OHLCV-only path is used.

    Parameters
    ----------
    session_factory:
        SQLAlchemy session factory backed by PostgreSQL.
    symbol:
        Instrument, e.g. ``"BTCUSDT"``.
    timeframe:
        Bar interval, e.g. ``"1"`` (Bybit 1-minute).
    source:
        Optional source filter for ``raw_bars`` and all public tables.
    feature_set_name:
        ``"v1"`` for OHLCV-only, ``"v1.6_public"`` for enriched.
    """
    enriched = feature_set_name == FEATURE_SET_PUBLIC

    with session_scope(session_factory) as session:
        market_repo = PostgresMarketDataRepository(session)
        bars = market_repo.fetch_raw_bars(symbol=symbol, timeframe=timeframe, source=source)

        if enriched:
            public_repo = PostgresPublicMarketRepository(session)
            mark_bars = public_repo.fetch_mark_bars(symbol=symbol, timeframe=timeframe)
            index_bars = public_repo.fetch_index_bars(symbol=symbol, timeframe=timeframe)
            funding = public_repo.fetch_funding(symbol=symbol, source=source)
            open_interest = public_repo.fetch_open_interest(symbol=symbol, source=source)
            long_short = public_repo.fetch_long_short_ratio(symbol=symbol, source=source)

    if bars.empty:
        raise ValueError(f"No raw bars found for {symbol} {timeframe}.")

    if enriched:
        dataset = build_dataset_from_public_tables(
            bars=bars,
            symbol=symbol,
            timeframe=timeframe,
            mark_bars=mark_bars if not mark_bars.empty else None,
            index_bars=index_bars if not index_bars.empty else None,
            funding=funding if not funding.empty else None,
            open_interest=open_interest if not open_interest.empty else None,
            long_short_ratio=long_short if not long_short.empty else None,
            feature_set_name=feature_set_name,
            label_horizon=3,
        )
    else:
        dataset = build_dataset_from_bars(
            bars=bars,
            symbol=symbol,
            timeframe=timeframe,
            feature_set_name=feature_set_name,
            label_horizon=3,
        )

    signals = generate_mean_reversion_signals(dataset, symbol=symbol)
    if signals.empty:
        raise ValueError("No signal rows could be generated from the dataset.")

    latest_signal = signals.iloc[-1]
    feature_row = dataset[dataset["event_time"] == latest_signal["event_time"]].iloc[-1]
    feature_snapshot = _feature_snapshot(feature_row)
    risk_checks = {
        "status": latest_signal["risk_status"],
        "reason": latest_signal["risk_rejection_reason"],
    }
    cost_assumptions = CostAssumptions().__dict__
    audit_id = hashlib.sha256(f"audit-{latest_signal['signal_id']}".encode()).hexdigest()[:24]
    audit = SignalAuditRow(
        audit_id=audit_id,
        signal_id=str(latest_signal["signal_id"]),
        symbol=symbol,
        event_time=latest_signal["event_time"],
        created_at=utc_now(),
        strategy_name=str(latest_signal["strategy_name"]),
        model_run_id=None,
        feature_set_version=feature_set_name,
        feature_snapshot_json=feature_snapshot,
        rationale_json=dict(latest_signal["rationale_json"]),
        risk_checks_json=risk_checks,
        cost_assumptions_json=cost_assumptions,
        decision=str(latest_signal["side"]),
        decision_reason="mean_reversion scaffold decision",
        system_mode="research",
        code_version="local",
        config_hash=hashlib.sha256(feature_set_name.encode()).hexdigest()[:16],
    )

    with session_scope(session_factory) as session:
        upsert_signals(session, _signal_rows_for_storage(signals))
        audit_repo = PostgresAuditRepository(session)
        audit_repo.upsert(audit)

    return ResearchPipelineResult(
        dataset_rows=len(dataset),
        signal_id=str(latest_signal["signal_id"]),
        audit_id=audit_id,
        decision=str(latest_signal["side"]),
    )
