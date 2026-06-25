from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import text

from app.paper.positions import apply_paper_close_to_positions, apply_paper_fill_to_positions
from app.storage.paper_read_repository import PaperReadRepository


class PaperWriteRepository(PaperReadRepository):
    def reset_demo_account(self, equity: float = 10_000.0) -> str:
        self._ensure_paper_intent_executions_table()
        self._ensure_paper_trade_closes_table()
        self._ensure_paper_lifecycle_events_table()
        self.session.execute(text("delete from paper_intent_executions"))
        self.session.execute(text("delete from paper_intents"))
        self.session.execute(text("delete from paper_trade_closes"))
        self.session.execute(text("delete from paper_lifecycle_events"))
        self.session.execute(text("delete from paper_fills"))
        self.session.execute(text("delete from paper_orders"))
        self.session.execute(text("delete from portfolio_snapshots"))
        snapshot_id = "demo-reset"
        self.session.execute(
            text("""
            insert into portfolio_snapshots (
                snapshot_id,
                event_time,
                equity,
                margin_used,
                available_margin,
                positions_json
            )
            values (
                :snapshot_id,
                :event_time,
                :equity,
                0,
                :equity,
                '{}'::jsonb
            )
            on conflict (snapshot_id) do update set
                event_time = excluded.event_time,
                equity = excluded.equity,
                margin_used = excluded.margin_used,
                available_margin = excluded.available_margin,
                positions_json = excluded.positions_json
            """),
            {
                "snapshot_id": snapshot_id,
                "event_time": datetime.now(UTC),
                "equity": equity,
            },
        )
        return snapshot_id

    def create_order_and_fill(
        self,
        order_id: str,
        fill_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        fee: float,
        created_at: datetime,
    ) -> None:
        self.session.execute(
            text("""
            insert into paper_orders (
                order_id,
                symbol,
                side,
                quantity,
                status,
                created_at
            )
            values (
                :order_id,
                :symbol,
                :side,
                :quantity,
                'filled',
                :created_at
            )
            on conflict (order_id) do update set
                symbol = excluded.symbol,
                side = excluded.side,
                quantity = excluded.quantity,
                status = excluded.status,
                created_at = excluded.created_at
            """),
            {
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "created_at": created_at,
            },
        )
        self.session.execute(
            text("""
            insert into paper_fills (
                fill_id,
                order_id,
                symbol,
                side,
                quantity,
                price,
                fee,
                filled_at
            )
            values (
                :fill_id,
                :order_id,
                :symbol,
                :side,
                :quantity,
                :price,
                :fee,
                :filled_at
            )
            on conflict (fill_id) do update set
                order_id = excluded.order_id,
                symbol = excluded.symbol,
                side = excluded.side,
                quantity = excluded.quantity,
                price = excluded.price,
                fee = excluded.fee,
                filled_at = excluded.filled_at
            """),
            {
                "fill_id": fill_id,
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "fee": fee,
                "filled_at": created_at,
            },
        )

    def upsert_paper_intent(
        self,
        *,
        intent_id: str,
        model_run_id: str,
        signal_id: str,
        symbol: str,
        side: str,
        event_time: datetime,
        status: str,
        approved_for_paper_demo: bool,
        created_at: datetime,
        intent_json: dict[str, Any],
    ) -> None:
        self._ensure_paper_intents_table()
        self.session.execute(
            text("""
            insert into paper_intents (
                intent_id,
                model_run_id,
                signal_id,
                symbol,
                side,
                event_time,
                status,
                approved_for_paper_demo,
                created_at,
                intent_json
            )
            values (
                :intent_id,
                :model_run_id,
                :signal_id,
                :symbol,
                :side,
                :event_time,
                :status,
                :approved_for_paper_demo,
                :created_at,
                cast(:intent_json as jsonb)
            )
            on conflict (intent_id) do update set
                model_run_id = excluded.model_run_id,
                signal_id = excluded.signal_id,
                symbol = excluded.symbol,
                side = excluded.side,
                event_time = excluded.event_time,
                status = excluded.status,
                approved_for_paper_demo = excluded.approved_for_paper_demo,
                created_at = excluded.created_at,
                intent_json = excluded.intent_json
            """),
            {
                "intent_id": intent_id,
                "model_run_id": model_run_id,
                "signal_id": signal_id,
                "symbol": symbol,
                "side": side,
                "event_time": event_time,
                "status": status,
                "approved_for_paper_demo": approved_for_paper_demo,
                "created_at": created_at,
                "intent_json": json.dumps(intent_json),
            },
        )

    def create_paper_trade_close(self, close_payload: dict[str, Any]) -> dict[str, Any]:
        self._ensure_paper_trade_closes_table()
        payload = dict(close_payload)
        entry_fill_id = str(payload.get("entry_fill_id", "")).strip()
        order_id = str(payload.get("order_id", "")).strip()
        symbol = str(payload.get("symbol", "")).strip()
        side = str(payload.get("side", "")).strip()
        close_reason = str(payload.get("close_reason", "manual_exit")).strip() or "manual_exit"
        closed_at_raw = payload.get("exit_time")
        if not entry_fill_id or not order_id or not symbol or not side:
            raise ValueError("Close payload is missing required routing fields.")
        if not isinstance(closed_at_raw, str):
            raise ValueError("Close payload exit_time must be ISO datetime string.")
        closed_at = datetime.fromisoformat(closed_at_raw.replace("Z", "+00:00"))
        close_id = str(payload.get("close_id", "")).strip()
        if not close_id:
            digest = hashlib.sha256(entry_fill_id.encode()).hexdigest()[:24]
            close_id = f"pclose_{digest}"
            payload["close_id"] = close_id
        self.session.execute(
            text("""
            insert into paper_trade_closes (
                close_id,
                entry_fill_id,
                order_id,
                symbol,
                side,
                quantity,
                entry_price,
                exit_price,
                gross_pnl,
                total_fees_paid,
                realized_pnl,
                close_reason,
                closed_at,
                close_json
            )
            values (
                :close_id,
                :entry_fill_id,
                :order_id,
                :symbol,
                :side,
                :quantity,
                :entry_price,
                :exit_price,
                :gross_pnl,
                :total_fees_paid,
                :realized_pnl,
                :close_reason,
                :closed_at,
                cast(:close_json as jsonb)
            )
            on conflict (entry_fill_id) do nothing
            """),
            {
                "close_id": close_id,
                "entry_fill_id": entry_fill_id,
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "quantity": float(payload["quantity"]),
                "entry_price": float(payload["entry_price"]),
                "exit_price": float(payload["exit_price"]),
                "gross_pnl": float(payload["gross_pnl"]),
                "total_fees_paid": float(payload["total_fees_paid"]),
                "realized_pnl": float(payload["realized_pnl"]),
                "close_reason": close_reason,
                "closed_at": closed_at,
                "close_json": json.dumps(payload),
            },
        )
        existing = self.get_paper_trade_close_by_fill(entry_fill_id)
        if existing is None:
            raise RuntimeError("Failed to persist paper trade close payload.")
        return existing

    def create_portfolio_snapshot_after_close(
        self, close_payload: dict[str, Any]
    ) -> dict[str, Any]:
        payload = dict(close_payload)
        close_id = str(payload.get("close_id", "")).strip()
        if not close_id:
            raise ValueError("close_id is required for portfolio snapshot update.")
        exit_time_raw = payload.get("exit_time")
        if not isinstance(exit_time_raw, str):
            raise ValueError("Close payload exit_time must be ISO datetime string.")
        exit_time = datetime.fromisoformat(exit_time_raw.replace("Z", "+00:00"))
        realized_pnl = float(payload["realized_pnl"])

        snapshot_id = self._portfolio_snapshot_id_for_close(close_id)
        existing_snapshot = self.get_portfolio_snapshot(snapshot_id)
        if existing_snapshot is not None:
            return existing_snapshot

        latest = self.latest_portfolio()
        previous_equity = float(latest["equity"])
        previous_available_margin = float(latest["available_margin"])
        margin_used = float(latest["margin_used"])
        positions = latest.get("positions", {})
        if not isinstance(positions, dict):
            positions = {}
        updated_positions = apply_paper_close_to_positions(
            current_positions=positions,
            close_payload=payload,
        )

        self.session.execute(
            text("""
            insert into portfolio_snapshots (
                snapshot_id,
                event_time,
                equity,
                margin_used,
                available_margin,
                positions_json
            )
            values (
                :snapshot_id,
                :event_time,
                :equity,
                :margin_used,
                :available_margin,
                cast(:positions_json as jsonb)
            )
            on conflict (snapshot_id) do nothing
            """),
            {
                "snapshot_id": snapshot_id,
                "event_time": exit_time,
                "equity": previous_equity + realized_pnl,
                "margin_used": margin_used,
                "available_margin": previous_available_margin + realized_pnl,
                "positions_json": json.dumps(updated_positions),
            },
        )

        created_snapshot = self.get_portfolio_snapshot(snapshot_id)
        if created_snapshot is not None:
            return created_snapshot
        return self.latest_portfolio()

    def create_portfolio_snapshot_after_fill(self, fill_payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(fill_payload)
        fill_id = str(payload.get("fill_id", "")).strip()
        if not fill_id:
            raise ValueError("fill_id is required for portfolio snapshot update.")
        filled_at_raw = payload.get("filled_at")
        if not isinstance(filled_at_raw, str):
            raise ValueError("Fill payload filled_at must be ISO datetime string.")
        filled_at = datetime.fromisoformat(filled_at_raw.replace("Z", "+00:00"))

        snapshot_id = self._portfolio_snapshot_id_for_fill(fill_id)
        existing_snapshot = self.get_portfolio_snapshot(snapshot_id)
        if existing_snapshot is not None:
            return existing_snapshot

        latest = self.latest_portfolio()
        equity = float(latest["equity"])
        available_margin = float(latest["available_margin"])
        margin_used = float(latest["margin_used"])
        positions = latest.get("positions", {})
        if not isinstance(positions, dict):
            positions = {}

        updated_positions = apply_paper_fill_to_positions(
            current_positions=positions,
            fill_payload=payload,
        )

        self.session.execute(
            text("""
            insert into portfolio_snapshots (
                snapshot_id,
                event_time,
                equity,
                margin_used,
                available_margin,
                positions_json
            )
            values (
                :snapshot_id,
                :event_time,
                :equity,
                :margin_used,
                :available_margin,
                cast(:positions_json as jsonb)
            )
            on conflict (snapshot_id) do nothing
            """),
            {
                "snapshot_id": snapshot_id,
                "event_time": filled_at,
                "equity": equity,
                "margin_used": margin_used,
                "available_margin": available_margin,
                "positions_json": json.dumps(updated_positions),
            },
        )

        created_snapshot = self.get_portfolio_snapshot(snapshot_id)
        if created_snapshot is not None:
            return created_snapshot
        return self.latest_portfolio()

    def upsert_paper_intent_execution(
        self,
        *,
        intent_id: str,
        order_id: str,
        fill_id: str,
        created_at: datetime,
    ) -> None:
        self._ensure_paper_intent_executions_table()
        self.session.execute(
            text("""
            insert into paper_intent_executions (
                intent_id,
                order_id,
                fill_id,
                created_at
            )
            values (
                :intent_id,
                :order_id,
                :fill_id,
                :created_at
            )
            on conflict (intent_id) do update set
                order_id = excluded.order_id,
                fill_id = excluded.fill_id,
                created_at = excluded.created_at
            """),
            {
                "intent_id": intent_id,
                "order_id": order_id,
                "fill_id": fill_id,
                "created_at": created_at,
            },
        )

    def append_paper_lifecycle_event(
        self,
        *,
        event_type: str,
        entity_type: str,
        entity_id: str,
        related_entity_id: str | None = None,
        created_at: datetime | None = None,
        event_json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._ensure_paper_lifecycle_events_table()
        normalized_event_type = event_type.strip()
        normalized_entity_type = entity_type.strip()
        normalized_entity_id = entity_id.strip()
        if not normalized_event_type:
            raise ValueError("event_type is required")
        if not normalized_entity_type:
            raise ValueError("entity_type is required")
        if not normalized_entity_id:
            raise ValueError("entity_id is required")

        timestamp = created_at or datetime.now(UTC)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        else:
            timestamp = timestamp.astimezone(UTC)
        payload = dict(event_json or {})
        event_id = f"plev_{uuid4().hex[:24]}"
        related_id = str(related_entity_id).strip() if related_entity_id is not None else None

        self.session.execute(
            text("""
            insert into paper_lifecycle_events (
                event_id,
                event_type,
                entity_type,
                entity_id,
                related_entity_id,
                created_at,
                event_json
            )
            values (
                :event_id,
                :event_type,
                :entity_type,
                :entity_id,
                :related_entity_id,
                :created_at,
                cast(:event_json as jsonb)
            )
            """),
            {
                "event_id": event_id,
                "event_type": normalized_event_type,
                "entity_type": normalized_entity_type,
                "entity_id": normalized_entity_id,
                "related_entity_id": related_id,
                "created_at": timestamp,
                "event_json": json.dumps(payload),
            },
        )
        return {
            "event_id": event_id,
            "event_type": normalized_event_type,
            "entity_type": normalized_entity_type,
            "entity_id": normalized_entity_id,
            "related_entity_id": related_id,
            "created_at": timestamp.isoformat(),
            "event_json": payload,
        }
