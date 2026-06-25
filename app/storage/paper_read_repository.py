from __future__ import annotations

import hashlib
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.paper.accounting import build_paper_fill_accounting


class PaperReadRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def _ensure_paper_intents_table(self) -> None:
        self.session.execute(text("""
            create table if not exists paper_intents (
                intent_id text primary key,
                model_run_id text not null,
                signal_id text not null,
                symbol text not null,
                side text not null,
                event_time timestamptz not null,
                status text not null,
                approved_for_paper_demo boolean not null,
                created_at timestamptz not null,
                intent_json jsonb not null
            )
            """))
        self.session.execute(text("""
            create index if not exists paper_intents_model_created_idx
            on paper_intents (model_run_id, created_at desc)
            """))
        self.session.execute(text("""
            create index if not exists paper_intents_symbol_created_idx
            on paper_intents (symbol, created_at desc)
            """))
        self.session.execute(text("""
            create index if not exists paper_intents_status_created_idx
            on paper_intents (status, created_at desc)
            """))

    def _ensure_paper_intent_executions_table(self) -> None:
        self._ensure_paper_intents_table()
        self.session.execute(text("""
            create table if not exists paper_intent_executions (
                intent_id text primary key references paper_intents(intent_id),
                order_id text not null references paper_orders(order_id),
                fill_id text not null references paper_fills(fill_id),
                created_at timestamptz not null
            )
            """))
        self.session.execute(text("""
            create index if not exists paper_intent_executions_created_idx
            on paper_intent_executions (created_at desc)
            """))

    def _ensure_paper_trade_closes_table(self) -> None:
        self.session.execute(text("""
            create table if not exists paper_trade_closes (
                close_id text primary key,
                entry_fill_id text not null unique references paper_fills(fill_id),
                order_id text not null references paper_orders(order_id),
                symbol text not null,
                side text not null,
                quantity numeric not null,
                entry_price numeric not null,
                exit_price numeric not null,
                gross_pnl numeric not null,
                total_fees_paid numeric not null,
                realized_pnl numeric not null,
                close_reason text not null,
                closed_at timestamptz not null,
                close_json jsonb not null
            )
            """))
        self.session.execute(text("""
            create index if not exists paper_trade_closes_entry_fill_idx
            on paper_trade_closes (entry_fill_id)
            """))
        self.session.execute(text("""
            create index if not exists paper_trade_closes_symbol_closed_idx
            on paper_trade_closes (symbol, closed_at desc)
            """))
        self.session.execute(text("""
            create index if not exists paper_trade_closes_closed_idx
            on paper_trade_closes (closed_at desc)
            """))

    def _ensure_paper_lifecycle_events_table(self) -> None:
        self.session.execute(text("""
            create table if not exists paper_lifecycle_events (
                event_id text primary key,
                event_type text not null,
                entity_type text not null,
                entity_id text not null,
                related_entity_id text,
                created_at timestamptz not null,
                event_json jsonb not null
            )
            """))
        self.session.execute(text("""
            create index if not exists paper_lifecycle_events_entity_idx
            on paper_lifecycle_events (entity_type, entity_id, created_at desc)
            """))
        self.session.execute(text("""
            create index if not exists paper_lifecycle_events_type_idx
            on paper_lifecycle_events (event_type, created_at desc)
            """))
        self.session.execute(text("""
            create index if not exists paper_lifecycle_events_created_idx
            on paper_lifecycle_events (created_at desc)
            """))

    def latest_portfolio(self) -> dict[str, Any]:
        row = self.session.execute(text("""
                select
                    snapshot_id,
                    event_time,
                    equity,
                    margin_used,
                    available_margin,
                    positions_json
                from portfolio_snapshots
                order by event_time desc
                limit 1
                """)).mappings().one_or_none()
        if row is None:
            return {
                "snapshot_id": None,
                "event_time": None,
                "equity": 10_000.0,
                "margin_used": 0.0,
                "available_margin": 10_000.0,
                "positions": {},
            }
        return {
            "snapshot_id": row["snapshot_id"],
            "event_time": row["event_time"].isoformat(),
            "equity": float(row["equity"]),
            "margin_used": float(row["margin_used"]),
            "available_margin": float(row["available_margin"]),
            "positions": dict(row["positions_json"]),
        }

    def _portfolio_snapshot_id_for_close(self, close_id: str) -> str:
        digest = hashlib.sha256(close_id.encode()).hexdigest()[:24]
        return f"psnap_close_{digest}"

    def _portfolio_snapshot_id_for_fill(self, fill_id: str) -> str:
        digest = hashlib.sha256(fill_id.encode()).hexdigest()[:24]
        return f"psnap_fill_{digest}"

    def get_portfolio_snapshot(self, snapshot_id: str) -> dict[str, Any] | None:
        row = (
            self.session.execute(
                text("""
                select
                    snapshot_id,
                    event_time,
                    equity,
                    margin_used,
                    available_margin,
                    positions_json
                from portfolio_snapshots
                where snapshot_id = :snapshot_id
                limit 1
                """),
                {"snapshot_id": snapshot_id},
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            return None
        return {
            "snapshot_id": row["snapshot_id"],
            "event_time": row["event_time"].isoformat(),
            "equity": float(row["equity"]),
            "margin_used": float(row["margin_used"]),
            "available_margin": float(row["available_margin"]),
            "positions": dict(row["positions_json"]),
        }

    def get_portfolio_snapshot_for_close(self, close_id: str) -> dict[str, Any] | None:
        snapshot_id = self._portfolio_snapshot_id_for_close(close_id)
        return self.get_portfolio_snapshot(snapshot_id)

    def get_portfolio_snapshot_for_fill(self, fill_id: str) -> dict[str, Any] | None:
        snapshot_id = self._portfolio_snapshot_id_for_fill(fill_id)
        return self.get_portfolio_snapshot(snapshot_id)

    def latest_portfolio_snapshots(self, limit: int = 500) -> list[dict[str, Any]]:
        rows = (
            self.session.execute(
                text("""
                select
                    snapshot_id,
                    event_time,
                    equity,
                    margin_used,
                    available_margin,
                    positions_json
                from portfolio_snapshots
                order by event_time desc
                limit :limit
                """),
                {"limit": limit},
            )
            .mappings()
            .all()
        )
        return [
            {
                "snapshot_id": row["snapshot_id"],
                "event_time": row["event_time"].isoformat(),
                "equity": float(row["equity"]),
                "margin_used": float(row["margin_used"]),
                "available_margin": float(row["available_margin"]),
                "positions": dict(row["positions_json"]),
            }
            for row in rows
        ]

    def latest_orders(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = (
            self.session.execute(
                text("""
                select order_id, symbol, side, quantity, status, created_at
                from paper_orders
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
                "order_id": str(row["order_id"]),
                "symbol": str(row["symbol"]),
                "side": str(row["side"]),
                "quantity": float(row["quantity"]),
                "status": str(row["status"]),
                "created_at": row["created_at"].isoformat(),
            }
            for row in rows
        ]

    def latest_paper_orders(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.latest_orders(limit=limit)

    def latest_fills(self, limit: int = 50) -> list[dict[str, Any]]:
        self._ensure_paper_trade_closes_table()
        rows = (
            self.session.execute(
                text("""
                select
                    f.fill_id,
                    f.order_id,
                    f.symbol,
                    f.side,
                    f.quantity,
                    f.price,
                    f.fee,
                    f.filled_at,
                    c.close_json
                from paper_fills f
                left join paper_trade_closes c
                    on c.entry_fill_id = f.fill_id
                order by f.filled_at desc
                limit :limit
                """),
                {"limit": limit},
            )
            .mappings()
            .all()
        )
        fills: list[dict[str, Any]] = []
        for row in rows:
            base_fill = {
                "fill_id": str(row["fill_id"]),
                "order_id": str(row["order_id"]),
                "symbol": str(row["symbol"]),
                "side": str(row["side"]),
                "quantity": float(row["quantity"]),
                "price": float(row["price"]),
                "fee": float(row["fee"]),
                "filled_at": row["filled_at"].isoformat(),
            }
            close_payload = row.get("close_json")
            source_payload = dict(close_payload) if isinstance(close_payload, dict) else None
            accounting = build_paper_fill_accounting(
                fill_payload=base_fill,
                source_payload=source_payload,
            )
            base_fill.update(accounting)
            fills.append(base_fill)
        return fills

    def latest_paper_intents(
        self,
        limit: int = 50,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        self._ensure_paper_intents_table()
        conditions: list[str] = []
        params: dict[str, Any] = {"limit": limit}
        if status is not None:
            conditions.append("status = :status")
            params["status"] = status
        where_sql = f"where {' and '.join(conditions)}" if conditions else ""
        rows = (
            self.session.execute(
                text(f"""
                select intent_json
                from paper_intents
                {where_sql}
                order by created_at desc
                limit :limit
                """),
                params,
            )
            .mappings()
            .all()
        )
        return [dict(row["intent_json"]) for row in rows]

    def get_paper_intent(self, intent_id: str) -> dict[str, Any] | None:
        self._ensure_paper_intents_table()
        row = (
            self.session.execute(
                text("""
                select intent_json
                from paper_intents
                where intent_id = :intent_id
                limit 1
                """),
                {"intent_id": intent_id},
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            return None
        return dict(row["intent_json"])

    def get_paper_order(self, order_id: str) -> dict[str, Any] | None:
        row = (
            self.session.execute(
                text("""
                select order_id, symbol, side, quantity, status, created_at
                from paper_orders
                where order_id = :order_id
                limit 1
                """),
                {"order_id": order_id},
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            return None
        return {
            "order_id": str(row["order_id"]),
            "symbol": str(row["symbol"]),
            "side": str(row["side"]),
            "quantity": float(row["quantity"]),
            "status": str(row["status"]),
            "created_at": row["created_at"].isoformat(),
        }

    def get_paper_fill(self, fill_id: str) -> dict[str, Any] | None:
        self._ensure_paper_trade_closes_table()
        row = (
            self.session.execute(
                text("""
                select
                    f.fill_id,
                    f.order_id,
                    f.symbol,
                    f.side,
                    f.quantity,
                    f.price,
                    f.fee,
                    f.filled_at,
                    c.close_json
                from paper_fills f
                left join paper_trade_closes c
                    on c.entry_fill_id = f.fill_id
                where f.fill_id = :fill_id
                limit 1
                """),
                {"fill_id": fill_id},
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            return None
        fill = {
            "fill_id": str(row["fill_id"]),
            "order_id": str(row["order_id"]),
            "symbol": str(row["symbol"]),
            "side": str(row["side"]),
            "quantity": float(row["quantity"]),
            "price": float(row["price"]),
            "fee": float(row["fee"]),
            "filled_at": row["filled_at"].isoformat(),
        }
        close_payload = row.get("close_json")
        source_payload = dict(close_payload) if isinstance(close_payload, dict) else None
        fill.update(build_paper_fill_accounting(fill_payload=fill, source_payload=source_payload))
        return fill

    def get_paper_trade_close_by_fill(self, fill_id: str) -> dict[str, Any] | None:
        self._ensure_paper_trade_closes_table()
        row = (
            self.session.execute(
                text("""
                select close_json
                from paper_trade_closes
                where entry_fill_id = :entry_fill_id
                limit 1
                """),
                {"entry_fill_id": fill_id},
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            return None
        return dict(row["close_json"])

    def latest_paper_trade_closes(self, limit: int = 100) -> list[dict[str, Any]]:
        self._ensure_paper_trade_closes_table()
        rows = (
            self.session.execute(
                text("""
                select close_json
                from paper_trade_closes
                order by closed_at desc
                limit :limit
                """),
                {"limit": limit},
            )
            .mappings()
            .all()
        )
        return [dict(row["close_json"]) for row in rows]

    def get_paper_intent_execution(self, intent_id: str) -> dict[str, Any] | None:
        self._ensure_paper_intent_executions_table()
        row = (
            self.session.execute(
                text("""
                select intent_id, order_id, fill_id, created_at
                from paper_intent_executions
                where intent_id = :intent_id
                limit 1
                """),
                {"intent_id": intent_id},
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            return None
        return {
            "intent_id": str(row["intent_id"]),
            "order_id": str(row["order_id"]),
            "fill_id": str(row["fill_id"]),
            "created_at": row["created_at"].isoformat(),
        }

    def latest_paper_intent_executions(self, limit: int = 100) -> list[dict[str, Any]]:
        self._ensure_paper_intent_executions_table()
        rows = (
            self.session.execute(
                text("""
                select intent_id, order_id, fill_id, created_at
                from paper_intent_executions
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
                "intent_id": str(row["intent_id"]),
                "order_id": str(row["order_id"]),
                "fill_id": str(row["fill_id"]),
                "created_at": row["created_at"].isoformat(),
            }
            for row in rows
        ]

    def latest_paper_lifecycle_events(
        self,
        *,
        limit: int = 100,
        event_type: str | None = None,
        entity_type: str | None = None,
        entity_id: str | None = None,
    ) -> list[dict[str, Any]]:
        self._ensure_paper_lifecycle_events_table()
        conditions: list[str] = []
        params: dict[str, Any] = {"limit": limit}
        if event_type is not None:
            conditions.append("event_type = :event_type")
            params["event_type"] = event_type
        if entity_type is not None:
            conditions.append("entity_type = :entity_type")
            params["entity_type"] = entity_type
        if entity_id is not None:
            conditions.append("entity_id = :entity_id")
            params["entity_id"] = entity_id
        where_sql = f"where {' and '.join(conditions)}" if conditions else ""
        rows = (
            self.session.execute(
                text(f"""
                select
                    event_id,
                    event_type,
                    entity_type,
                    entity_id,
                    related_entity_id,
                    created_at,
                    event_json
                from paper_lifecycle_events
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
                "event_id": str(row["event_id"]),
                "event_type": str(row["event_type"]),
                "entity_type": str(row["entity_type"]),
                "entity_id": str(row["entity_id"]),
                "related_entity_id": (
                    str(row["related_entity_id"]) if row["related_entity_id"] is not None else None
                ),
                "created_at": row["created_at"].isoformat(),
                "event_json": dict(row["event_json"]),
            }
            for row in rows
        ]
