from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.storage.models import SignalAuditRow
from app.storage.paper_write_repository import PaperWriteRepository


class InMemoryAuditRepository:
    def __init__(self) -> None:
        self.rows: list[SignalAuditRow] = []

    def add(self, row: SignalAuditRow) -> None:
        self.rows.append(row)

    def latest(self) -> SignalAuditRow | None:
        return self.rows[-1] if self.rows else None


class PostgresAuditRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert(self, row: SignalAuditRow) -> None:
        self.session.execute(
            text("""
                insert into signal_audit_logs (
                    audit_id,
                    signal_id,
                    symbol,
                    event_time,
                    created_at,
                    strategy_name,
                    model_run_id,
                    feature_set_version,
                    feature_snapshot_json,
                    rationale_json,
                    risk_checks_json,
                    cost_assumptions_json,
                    decision,
                    decision_reason,
                    system_mode,
                    code_version,
                    config_hash
                )
                values (
                    :audit_id,
                    :signal_id,
                    :symbol,
                    :event_time,
                    :created_at,
                    :strategy_name,
                    :model_run_id,
                    :feature_set_version,
                    cast(:feature_snapshot_json as jsonb),
                    cast(:rationale_json as jsonb),
                    cast(:risk_checks_json as jsonb),
                    cast(:cost_assumptions_json as jsonb),
                    :decision,
                    :decision_reason,
                    :system_mode,
                    :code_version,
                    :config_hash
                )
                on conflict (audit_id) do update set
                    signal_id = excluded.signal_id,
                    symbol = excluded.symbol,
                    event_time = excluded.event_time,
                    created_at = excluded.created_at,
                    strategy_name = excluded.strategy_name,
                    model_run_id = excluded.model_run_id,
                    feature_set_version = excluded.feature_set_version,
                    feature_snapshot_json = excluded.feature_snapshot_json,
                    rationale_json = excluded.rationale_json,
                    risk_checks_json = excluded.risk_checks_json,
                    cost_assumptions_json = excluded.cost_assumptions_json,
                    decision = excluded.decision,
                    decision_reason = excluded.decision_reason,
                    system_mode = excluded.system_mode,
                    code_version = excluded.code_version,
                    config_hash = excluded.config_hash
                """),
            {
                "audit_id": row.audit_id,
                "signal_id": row.signal_id,
                "symbol": row.symbol,
                "event_time": row.event_time,
                "created_at": row.created_at,
                "strategy_name": row.strategy_name,
                "model_run_id": row.model_run_id,
                "feature_set_version": row.feature_set_version,
                "feature_snapshot_json": json.dumps(row.feature_snapshot_json),
                "rationale_json": json.dumps(row.rationale_json),
                "risk_checks_json": json.dumps(row.risk_checks_json),
                "cost_assumptions_json": json.dumps(row.cost_assumptions_json),
                "decision": row.decision,
                "decision_reason": row.decision_reason,
                "system_mode": row.system_mode,
                "code_version": row.code_version,
                "config_hash": row.config_hash,
            },
        )

    def get(self, audit_id: str) -> SignalAuditRow | None:
        result = (
            self.session.execute(
                text("""
                select
                    audit_id,
                    signal_id,
                    symbol,
                    event_time,
                    created_at,
                    strategy_name,
                    model_run_id,
                    feature_set_version,
                    feature_snapshot_json,
                    rationale_json,
                    risk_checks_json,
                    cost_assumptions_json,
                    decision,
                    decision_reason,
                    system_mode,
                    code_version,
                    config_hash
                from signal_audit_logs
                where audit_id = :audit_id
                """),
                {"audit_id": audit_id},
            )
            .mappings()
            .one_or_none()
        )
        if result is None:
            return None
        return SignalAuditRow(
            audit_id=str(result["audit_id"]),
            signal_id=str(result["signal_id"]),
            symbol=str(result["symbol"]),
            event_time=result["event_time"],
            created_at=result["created_at"],
            strategy_name=str(result["strategy_name"]),
            model_run_id=result["model_run_id"],
            feature_set_version=str(result["feature_set_version"]),
            feature_snapshot_json=dict(result["feature_snapshot_json"]),
            rationale_json=dict(result["rationale_json"]),
            risk_checks_json=dict(result["risk_checks_json"]),
            cost_assumptions_json=dict(result["cost_assumptions_json"]),
            decision=str(result["decision"]),
            decision_reason=str(result["decision_reason"]),
            system_mode=str(result["system_mode"]),
            code_version=str(result["code_version"]),
            config_hash=str(result["config_hash"]),
        )

    def get_by_signal_id(self, signal_id: str) -> SignalAuditRow | None:
        result = (
            self.session.execute(
                text("""
                select
                    audit_id,
                    signal_id,
                    symbol,
                    event_time,
                    created_at,
                    strategy_name,
                    model_run_id,
                    feature_set_version,
                    feature_snapshot_json,
                    rationale_json,
                    risk_checks_json,
                    cost_assumptions_json,
                    decision,
                    decision_reason,
                    system_mode,
                    code_version,
                    config_hash
                from signal_audit_logs
                where signal_id = :signal_id
                order by created_at desc
                limit 1
                """),
                {"signal_id": signal_id},
            )
            .mappings()
            .one_or_none()
        )
        if result is None:
            return None
        return SignalAuditRow(
            audit_id=str(result["audit_id"]),
            signal_id=str(result["signal_id"]),
            symbol=str(result["symbol"]),
            event_time=result["event_time"],
            created_at=result["created_at"],
            strategy_name=str(result["strategy_name"]),
            model_run_id=result["model_run_id"],
            feature_set_version=str(result["feature_set_version"]),
            feature_snapshot_json=dict(result["feature_snapshot_json"]),
            rationale_json=dict(result["rationale_json"]),
            risk_checks_json=dict(result["risk_checks_json"]),
            cost_assumptions_json=dict(result["cost_assumptions_json"]),
            decision=str(result["decision"]),
            decision_reason=str(result["decision_reason"]),
            system_mode=str(result["system_mode"]),
            code_version=str(result["code_version"]),
            config_hash=str(result["config_hash"]),
        )


class PostgresMarketDataRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def fetch_raw_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        source: str | None = None,
    ) -> pd.DataFrame:
        conditions = ["symbol = :symbol", "timeframe = :timeframe"]
        params: dict[str, Any] = {"symbol": symbol, "timeframe": timeframe}
        if start is not None:
            conditions.append("event_time >= :start")
            params["start"] = start
        if end is not None:
            conditions.append("event_time <= :end")
            params["end"] = end
        if source is not None:
            conditions.append("source = :source")
            params["source"] = source

        result = (
            self.session.execute(
                text(f"""
                select
                    event_time,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    fetch_time,
                    source
                from raw_bars
                where {" and ".join(conditions)}
                order by event_time
                """),
                params,
            )
            .mappings()
            .all()
        )
        frame = pd.DataFrame(result)
        if frame.empty:
            return frame
        for column in ("open", "high", "low", "close", "volume"):
            frame[column] = frame[column].astype(float)
        return frame

    def latest_raw_bars(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 200,
        source: str | None = None,
    ) -> list[dict[str, Any]]:
        conditions = ["symbol = :symbol", "timeframe = :timeframe"]
        params: dict[str, Any] = {"symbol": symbol, "timeframe": timeframe, "limit": limit}
        if source is not None:
            conditions.append("source = :source")
            params["source"] = source

        result = (
            self.session.execute(
                text(f"""
                select
                    event_time,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    source
                from raw_bars
                where {" and ".join(conditions)}
                order by event_time desc
                limit :limit
                """),
                params,
            )
            .mappings()
            .all()
        )
        rows = [
            {
                "event_time": row["event_time"].isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "source": str(row["source"]),
            }
            for row in result
        ]
        return list(reversed(rows))


class PostgresSignalRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def latest(self, symbol: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        conditions: list[str] = []
        params: dict[str, Any] = {"limit": limit}
        if symbol is not None:
            conditions.append("symbol = :symbol")
            params["symbol"] = symbol
        where_sql = f"where {' and '.join(conditions)}" if conditions else ""
        result = (
            self.session.execute(
                text(f"""
                select
                    signal_id,
                    symbol,
                    event_time,
                    strategy_name,
                    side,
                    score,
                    confidence,
                    rationale_json
                from signals
                {where_sql}
                order by event_time desc
                limit :limit
                """),
                params,
            )
            .mappings()
            .all()
        )
        return [
            {
                "signal_id": str(row["signal_id"]),
                "symbol": str(row["symbol"]),
                "event_time": row["event_time"].isoformat(),
                "strategy_name": str(row["strategy_name"]),
                "side": str(row["side"]),
                "score": float(row["score"]),
                "confidence": float(row["confidence"]),
                "rationale_json": dict(row["rationale_json"]),
            }
            for row in result
        ]


class PostgresBacktestRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_run(
        self,
        run_id: str,
        created_at: datetime,
        config_json: dict[str, Any],
        metrics_json: dict[str, Any],
    ) -> None:
        self.session.execute(
            text("""
            insert into backtest_runs (
                run_id,
                created_at,
                config_json,
                metrics_json
            )
            values (
                :run_id,
                :created_at,
                cast(:config_json as jsonb),
                cast(:metrics_json as jsonb)
            )
            on conflict (run_id) do update set
                created_at = excluded.created_at,
                config_json = excluded.config_json,
                metrics_json = excluded.metrics_json
            """),
            {
                "run_id": run_id,
                "created_at": created_at,
                "config_json": json.dumps(config_json),
                "metrics_json": json.dumps(metrics_json),
            },
        )

    def upsert_trades(self, trades: list[dict[str, Any]]) -> int:
        if not trades:
            return 0
        self.session.execute(
            text("""
            insert into backtest_trades (
                trade_id,
                run_id,
                symbol,
                side,
                entry_time,
                exit_time,
                pnl
            )
            values (
                :trade_id,
                :run_id,
                :symbol,
                :side,
                :entry_time,
                :exit_time,
                :pnl
            )
            on conflict (trade_id) do update set
                run_id = excluded.run_id,
                symbol = excluded.symbol,
                side = excluded.side,
                entry_time = excluded.entry_time,
                exit_time = excluded.exit_time,
                pnl = excluded.pnl
            """),
            trades,
        )
        return len(trades)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        run = (
            self.session.execute(
                text("""
                select run_id, created_at, config_json, metrics_json
                from backtest_runs
                where run_id = :run_id
                """),
                {"run_id": run_id},
            )
            .mappings()
            .one_or_none()
        )
        if run is None:
            return None
        trades = (
            self.session.execute(
                text("""
                select trade_id, symbol, side, entry_time, exit_time, pnl
                from backtest_trades
                where run_id = :run_id
                order by entry_time
                """),
                {"run_id": run_id},
            )
            .mappings()
            .all()
        )
        return {
            "run_id": str(run["run_id"]),
            "created_at": run["created_at"].isoformat(),
            "config": dict(run["config_json"]),
            "metrics": dict(run["metrics_json"]),
            "trades": [
                {
                    "trade_id": str(trade["trade_id"]),
                    "symbol": str(trade["symbol"]),
                    "side": str(trade["side"]),
                    "entry_time": trade["entry_time"].isoformat(),
                    "exit_time": (
                        trade["exit_time"].isoformat() if trade["exit_time"] is not None else None
                    ),
                    "pnl": float(trade["pnl"]),
                }
                for trade in trades
            ],
        }

    def latest_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = (
            self.session.execute(
                text("""
                select run_id, created_at, config_json, metrics_json
                from backtest_runs
                order by created_at desc
                limit :limit
                """),
                {"limit": limit},
            )
            .mappings()
            .all()
        )
        return [
            {
                "run_id": str(row["run_id"]),
                "created_at": row["created_at"].isoformat(),
                "config": dict(row["config_json"]),
                "metrics": dict(row["metrics_json"]),
            }
            for row in rows
        ]


class PostgresPaperRepository(PaperWriteRepository):
    """Compatibility facade for paper read/write repository APIs."""


class PostgresPublicMarketRepository:
    """Read-side repository for the five public Bybit market tables.

    All methods enforce the no-lookahead rule: data is fetched only at or
    before ``end`` (i.e. ``event_time <= end``).  Sparse tables (funding,
    OI, long-short) return the most-recent known row per symbol/timeframe
    bucket when joined to a bar frame.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    # ------------------------------------------------------------------
    # Mark-price bars
    # ------------------------------------------------------------------

    def fetch_mark_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        conditions = ["symbol = :symbol", "timeframe = :timeframe"]
        params: dict[str, Any] = {"symbol": symbol, "timeframe": timeframe}
        if start is not None:
            conditions.append("event_time >= :start")
            params["start"] = start
        if end is not None:
            conditions.append("event_time <= :end")
            params["end"] = end
        result = (
            self.session.execute(
                text(f"""
                select event_time, open, high, low, close, volume
                from raw_mark_bars
                where {" and ".join(conditions)}
                order by event_time
                """),
                params,
            )
            .mappings()
            .all()
        )
        frame = pd.DataFrame(result)
        if frame.empty:
            return frame
        for col in ("open", "high", "low", "close", "volume"):
            frame[col] = frame[col].astype(float)
        return frame

    # ------------------------------------------------------------------
    # Index-price bars
    # ------------------------------------------------------------------

    def fetch_index_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        conditions = ["symbol = :symbol", "timeframe = :timeframe"]
        params: dict[str, Any] = {"symbol": symbol, "timeframe": timeframe}
        if start is not None:
            conditions.append("event_time >= :start")
            params["start"] = start
        if end is not None:
            conditions.append("event_time <= :end")
            params["end"] = end
        result = (
            self.session.execute(
                text(f"""
                select event_time, open, high, low, close, volume
                from raw_index_bars
                where {" and ".join(conditions)}
                order by event_time
                """),
                params,
            )
            .mappings()
            .all()
        )
        frame = pd.DataFrame(result)
        if frame.empty:
            return frame
        for col in ("open", "high", "low", "close", "volume"):
            frame[col] = frame[col].astype(float)
        return frame

    # ------------------------------------------------------------------
    # Funding history  (sparse — one row every 8 h)
    # ------------------------------------------------------------------

    def fetch_funding(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
        source: str | None = None,
    ) -> pd.DataFrame:
        """Return all funding events in [start, end].

        Rows are ordered by ``event_time``.  Callers must use
        ``merge_asof`` (or equivalent) to avoid lookahead when aligning to
        bar timestamps.  Pass ``source`` to restrict to a specific ingestion
        source and avoid mixing rows from different pipelines.
        """
        conditions = ["symbol = :symbol"]
        params: dict[str, Any] = {"symbol": symbol}
        if start is not None:
            conditions.append("event_time >= :start")
            params["start"] = start
        if end is not None:
            conditions.append("event_time <= :end")
            params["end"] = end
        if source is not None:
            conditions.append("source = :source")
            params["source"] = source
        result = (
            self.session.execute(
                text(f"""
                select event_time, funding_rate
                from raw_funding
                where {" and ".join(conditions)}
                order by event_time
                """),
                params,
            )
            .mappings()
            .all()
        )
        frame = pd.DataFrame(result)
        if frame.empty:
            return frame
        frame["funding_rate"] = frame["funding_rate"].astype(float)
        return frame

    # ------------------------------------------------------------------
    # Open Interest  (sparse — one row every 5–15 m typically)
    # ------------------------------------------------------------------

    def fetch_open_interest(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
        source: str | None = None,
    ) -> pd.DataFrame:
        """Return all OI snapshots in [start, end], ordered by event_time.

        Pass ``source`` to restrict to a specific ingestion source.
        """
        conditions = ["symbol = :symbol"]
        params: dict[str, Any] = {"symbol": symbol}
        if start is not None:
            conditions.append("event_time >= :start")
            params["start"] = start
        if end is not None:
            conditions.append("event_time <= :end")
            params["end"] = end
        if source is not None:
            conditions.append("source = :source")
            params["source"] = source
        result = (
            self.session.execute(
                text(f"""
                select event_time, open_interest
                from raw_open_interest
                where {" and ".join(conditions)}
                order by event_time
                """),
                params,
            )
            .mappings()
            .all()
        )
        frame = pd.DataFrame(result)
        if frame.empty:
            return frame
        frame["open_interest"] = frame["open_interest"].astype(float)
        return frame

    # ------------------------------------------------------------------
    # Long-short ratio  (sparse — one row every 5–15 m typically)
    # ------------------------------------------------------------------

    def fetch_long_short_ratio(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
        source: str | None = None,
    ) -> pd.DataFrame:
        """Return all long-short ratio snapshots in [start, end], ordered by event_time.

        Pass ``source`` to restrict to a specific ingestion source.
        """
        conditions = ["symbol = :symbol"]
        params: dict[str, Any] = {"symbol": symbol}
        if start is not None:
            conditions.append("event_time >= :start")
            params["start"] = start
        if end is not None:
            conditions.append("event_time <= :end")
            params["end"] = end
        if source is not None:
            conditions.append("source = :source")
            params["source"] = source
        result = (
            self.session.execute(
                text(f"""
                select event_time, long_short_ratio
                from raw_long_short_ratio
                where {" and ".join(conditions)}
                order by event_time
                """),
                params,
            )
            .mappings()
            .all()
        )
        frame = pd.DataFrame(result)
        if frame.empty:
            return frame
        frame["long_short_ratio"] = frame["long_short_ratio"].astype(float)
        return frame


class PostgresModelRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def _ensure_strategy_comparison_snapshot_table(self) -> None:
        self.session.execute(text("""
            create table if not exists strategy_comparison_snapshots (
                snapshot_id text primary key,
                model_run_id text not null,
                symbol text not null,
                timeframe text not null,
                feature_set_version text not null,
                created_at timestamptz not null,
                comparison_json jsonb not null
            )
            """))
        self.session.execute(text("""
            create index if not exists strategy_comparison_snapshots_model_created_idx
            on strategy_comparison_snapshots (model_run_id, created_at desc)
            """))
        self.session.execute(text("""
            create index if not exists strategy_comparison_snapshots_symbol_tf_created_idx
            on strategy_comparison_snapshots (symbol, timeframe, created_at desc)
            """))

    def upsert_model_run(
        self,
        model_run_id: str,
        symbol: str,
        created_at: datetime,
        feature_set_version: str,
        metrics_json: dict[str, Any],
        params_json: dict[str, Any],
    ) -> None:
        self.session.execute(
            text("""
            insert into model_runs (
                model_run_id,
                symbol,
                created_at,
                feature_set_version,
                metrics_json,
                params_json
            )
            values (
                :model_run_id,
                :symbol,
                :created_at,
                :feature_set_version,
                cast(:metrics_json as jsonb),
                cast(:params_json as jsonb)
            )
            on conflict (model_run_id) do update set
                symbol = excluded.symbol,
                created_at = excluded.created_at,
                feature_set_version = excluded.feature_set_version,
                metrics_json = excluded.metrics_json,
                params_json = excluded.params_json
            """),
            {
                "model_run_id": model_run_id,
                "symbol": symbol,
                "created_at": created_at,
                "feature_set_version": feature_set_version,
                "metrics_json": json.dumps(metrics_json),
                "params_json": json.dumps(params_json),
            },
        )

    def upsert_strategy_comparison_snapshot(
        self,
        snapshot_id: str,
        model_run_id: str,
        symbol: str,
        timeframe: str,
        feature_set_version: str,
        created_at: datetime,
        comparison_json: dict[str, Any],
    ) -> None:
        self._ensure_strategy_comparison_snapshot_table()
        self.session.execute(
            text("""
            insert into strategy_comparison_snapshots (
                snapshot_id,
                model_run_id,
                symbol,
                timeframe,
                feature_set_version,
                created_at,
                comparison_json
            )
            values (
                :snapshot_id,
                :model_run_id,
                :symbol,
                :timeframe,
                :feature_set_version,
                :created_at,
                cast(:comparison_json as jsonb)
            )
            on conflict (snapshot_id) do update set
                model_run_id = excluded.model_run_id,
                symbol = excluded.symbol,
                timeframe = excluded.timeframe,
                feature_set_version = excluded.feature_set_version,
                created_at = excluded.created_at,
                comparison_json = excluded.comparison_json
            """),
            {
                "snapshot_id": snapshot_id,
                "model_run_id": model_run_id,
                "symbol": symbol,
                "timeframe": timeframe,
                "feature_set_version": feature_set_version,
                "created_at": created_at,
                "comparison_json": json.dumps(comparison_json),
            },
        )

    def latest_strategy_comparison_snapshot(self, model_run_id: str) -> dict[str, Any] | None:
        self._ensure_strategy_comparison_snapshot_table()
        row = (
            self.session.execute(
                text("""
                select
                    snapshot_id,
                    model_run_id,
                    symbol,
                    timeframe,
                    feature_set_version,
                    created_at,
                    comparison_json
                from strategy_comparison_snapshots
                where model_run_id = :model_run_id
                order by created_at desc
                limit 1
                """),
                {"model_run_id": model_run_id},
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            return None
        return {
            "snapshot_id": str(row["snapshot_id"]),
            "model_run_id": str(row["model_run_id"]),
            "symbol": str(row["symbol"]),
            "timeframe": str(row["timeframe"]),
            "feature_set_version": str(row["feature_set_version"]),
            "created_at": row["created_at"].isoformat(),
            "comparison_json": dict(row["comparison_json"]),
        }

    def latest(self, symbol: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        conditions: list[str] = []
        params: dict[str, Any] = {"limit": limit}
        if symbol is not None:
            conditions.append("symbol = :symbol")
            params["symbol"] = symbol
        where_sql = f"where {' and '.join(conditions)}" if conditions else ""
        rows = (
            self.session.execute(
                text(f"""
                select
                    model_run_id,
                    symbol,
                    created_at,
                    feature_set_version,
                    metrics_json,
                    params_json
                from model_runs
                {where_sql}
                order by created_at desc
                limit :limit
                """),
                params,
            )
            .mappings()
            .all()
        )
        return [
            {
                "model_run_id": str(row["model_run_id"]),
                "symbol": str(row["symbol"]),
                "created_at": row["created_at"].isoformat(),
                "feature_set_version": str(row["feature_set_version"]),
                "metrics": dict(row["metrics_json"]),
                "params": dict(row["params_json"]),
            }
            for row in rows
        ]
