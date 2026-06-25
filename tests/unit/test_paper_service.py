from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from app.paper.service import (
    PaperServiceError,
    close_paper_fill_service,
    persist_paper_intent_service,
    simulate_order_from_intent_service,
)


class _FakePaperRepo:
    def __init__(self) -> None:
        self.intents: dict[str, dict[str, Any]] = {}
        self.executions: dict[str, dict[str, Any]] = {}
        self.orders: dict[str, dict[str, Any]] = {}
        self.fills: dict[str, dict[str, Any]] = {}
        self.closes_by_fill: dict[str, dict[str, Any]] = {}
        self.snapshots_for_fill: dict[str, dict[str, Any]] = {}
        self.snapshots_for_close: dict[str, dict[str, Any]] = {}
        self.latest_portfolio_value: dict[str, Any] = {
            "snapshot_id": "latest",
            "equity": 10_000.0,
            "available_margin": 10_000.0,
            "margin_used": 0.0,
            "positions": {},
        }
        self.last_upsert_intent: dict[str, Any] | None = None
        self.lifecycle_events: list[dict[str, Any]] = []

    def get_paper_intent(self, intent_id: str) -> dict[str, Any] | None:
        return self.intents.get(intent_id)

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
        self.last_upsert_intent = {
            "intent_id": intent_id,
            "model_run_id": model_run_id,
            "signal_id": signal_id,
            "symbol": symbol,
            "side": side,
            "event_time": event_time.isoformat(),
            "status": status,
            "approved_for_paper_demo": approved_for_paper_demo,
            "created_at": created_at.isoformat(),
            "intent_json": dict(intent_json),
        }

    def get_paper_intent_execution(self, intent_id: str) -> dict[str, Any] | None:
        return self.executions.get(intent_id)

    def get_paper_order(self, order_id: str) -> dict[str, Any] | None:
        return self.orders.get(order_id)

    def get_paper_fill(self, fill_id: str) -> dict[str, Any] | None:
        return self.fills.get(fill_id)

    def get_portfolio_snapshot_for_fill(self, fill_id: str) -> dict[str, Any] | None:
        return self.snapshots_for_fill.get(fill_id)

    def latest_portfolio(self) -> dict[str, Any]:
        return dict(self.latest_portfolio_value)

    def create_order_and_fill(
        self,
        *,
        order_id: str,
        fill_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        fee: float,
        created_at: datetime,
    ) -> None:
        self.orders[order_id] = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "status": "filled",
            "created_at": created_at.isoformat(),
        }
        self.fills[fill_id] = {
            "fill_id": fill_id,
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "fee": fee,
            "filled_at": created_at.isoformat(),
            "entry_price": price,
            "entry_time": created_at.isoformat(),
            "pnl_status": "open",
            "realized_pnl": None,
            "unrealized_pnl": None,
        }

    def upsert_paper_intent_execution(
        self,
        *,
        intent_id: str,
        order_id: str,
        fill_id: str,
        created_at: datetime,
    ) -> None:
        self.executions[intent_id] = {
            "intent_id": intent_id,
            "order_id": order_id,
            "fill_id": fill_id,
            "created_at": created_at.isoformat(),
        }

    def create_portfolio_snapshot_after_fill(self, fill_payload: dict[str, Any]) -> dict[str, Any]:
        fill_id = str(fill_payload["fill_id"])
        snapshot = {
            "snapshot_id": f"psnap_fill_{fill_id}",
            "equity": 10_000.0,
            "available_margin": 10_000.0,
            "margin_used": 0.0,
            "positions": {
                str(fill_payload["symbol"]): {
                    "symbol": str(fill_payload["symbol"]),
                    "side": "long" if "long" in str(fill_payload["side"]) else "short",
                    "status": "open",
                    "entry_fill_id": fill_id,
                }
            },
        }
        self.snapshots_for_fill[fill_id] = snapshot
        self.latest_portfolio_value = snapshot
        return snapshot

    def get_paper_trade_close_by_fill(self, fill_id: str) -> dict[str, Any] | None:
        return self.closes_by_fill.get(fill_id)

    def get_portfolio_snapshot_for_close(self, close_id: str) -> dict[str, Any] | None:
        return self.snapshots_for_close.get(close_id)

    def create_paper_trade_close(self, close_payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(close_payload)
        payload.setdefault("close_id", "pclose_test_001")
        self.closes_by_fill[str(payload["entry_fill_id"])] = payload
        return payload

    def create_portfolio_snapshot_after_close(
        self, close_payload: dict[str, Any]
    ) -> dict[str, Any]:
        close_id = str(close_payload["close_id"])
        snapshot = {
            "snapshot_id": f"psnap_close_{close_id}",
            "equity": 10_005.0,
            "available_margin": 10_005.0,
            "margin_used": 0.0,
            "positions": {},
        }
        self.snapshots_for_close[close_id] = snapshot
        self.latest_portfolio_value = snapshot
        return snapshot

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
        event = {
            "event_type": event_type,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "related_entity_id": related_entity_id,
            "created_at": (
                created_at.isoformat()
                if isinstance(created_at, datetime)
                else datetime.now(UTC).isoformat()
            ),
            "event_json": dict(event_json or {}),
        }
        self.lifecycle_events.append(event)
        return event


def _candidate_intent(intent_id: str = "intent_001") -> dict[str, Any]:
    return {
        "intent_id": intent_id,
        "model_run_id": "model_001",
        "signal_id": "signal_001",
        "symbol": "BTCUSDT",
        "side": "long",
        "status": "candidate",
        "approved_for_paper_demo": True,
        "source_signal": {
            "entry_ref": 100.0,
            "stop_ref": 95.0,
            "take_ref": 110.0,
            "ttl_sec": 1800,
        },
        "risk_preview": {},
        "warnings": [],
    }


def _open_fill(fill_id: str = "fill_001") -> dict[str, Any]:
    return {
        "fill_id": fill_id,
        "order_id": "order_001",
        "symbol": "BTCUSDT",
        "side": "buy_open_long",
        "quantity": 1.0,
        "price": 100.0,
        "fee": 0.055,
        "filled_at": "2026-01-01T00:00:00+00:00",
        "entry_price": 100.0,
        "entry_time": "2026-01-01T00:00:00+00:00",
        "pnl_status": "open",
        "realized_pnl": None,
        "unrealized_pnl": None,
    }


def _assert_manual_simulation_flags(payload: dict[str, object]) -> None:
    assert payload["simulation_only"] is True
    assert payload["manual_trigger"] is True
    assert payload["real_execution"] is False


def test_persist_paper_intent_service_happy_path() -> None:
    repo = _FakePaperRepo()
    intent_payload = {
        "intent_id": "intent_persist_001",
        "model_run_id": "model_001",
        "signal_id": "signal_001",
        "symbol": "BTCUSDT",
        "side": "long",
        "event_time": "2026-01-01T00:00:00+00:00",
        "status": "candidate",
        "approved_for_paper_demo": True,
        "created_at": "2026-01-01T00:00:01+00:00",
    }

    payload = persist_paper_intent_service(repo=repo, intent_payload=intent_payload)
    assert payload == intent_payload
    assert repo.last_upsert_intent is not None
    assert repo.last_upsert_intent["intent_id"] == "intent_persist_001"
    assert repo.last_upsert_intent["event_time"] == "2026-01-01T00:00:00+00:00"
    assert repo.last_upsert_intent["created_at"] == "2026-01-01T00:00:01+00:00"
    assert repo.lifecycle_events[-1]["event_type"] == "intent_created"
    assert repo.lifecycle_events[-1]["entity_type"] == "paper_intent"
    assert repo.lifecycle_events[-1]["entity_id"] == "intent_persist_001"


def test_persist_paper_intent_service_invalid_timestamps_raise_400() -> None:
    repo = _FakePaperRepo()
    intent_payload = {
        "intent_id": "intent_persist_invalid",
        "model_run_id": "model_001",
        "signal_id": "signal_001",
        "symbol": "BTCUSDT",
        "side": "long",
        "event_time": "invalid",
        "status": "candidate",
        "approved_for_paper_demo": True,
        "created_at": "2026-01-01T00:00:01+00:00",
    }

    with pytest.raises(PaperServiceError) as exc:
        persist_paper_intent_service(repo=repo, intent_payload=intent_payload)
    assert exc.value.status_code == 400
    assert str(exc.value.detail).startswith("Invalid intent timestamps:")


def test_simulate_order_service_happy_path() -> None:
    repo = _FakePaperRepo()
    repo.intents["intent_001"] = _candidate_intent()

    payload = simulate_order_from_intent_service(
        repo=repo,
        intent_id="intent_001",
        execution_config={},
    )

    _assert_manual_simulation_flags(payload)
    assert payload["intent_id"] == "intent_001"
    assert payload["source_intent_id"] == "intent_001"
    assert isinstance(payload["order"], dict)
    assert isinstance(payload["fill"], dict)
    assert isinstance(payload["portfolio_snapshot"], dict)
    assert repo.lifecycle_events[-1]["event_type"] == "intent_simulated"
    assert repo.lifecycle_events[-1]["entity_type"] == "paper_intent"
    assert repo.lifecycle_events[-1]["entity_id"] == "intent_001"


def test_simulate_order_service_missing_intent_raises_404() -> None:
    repo = _FakePaperRepo()
    with pytest.raises(PaperServiceError) as exc:
        simulate_order_from_intent_service(
            repo=repo,
            intent_id="missing",
            execution_config={},
        )
    assert exc.value.status_code == 404


def test_simulate_order_service_blocked_order_request_raises_409() -> None:
    repo = _FakePaperRepo()
    blocked = _candidate_intent("intent_blocked")
    blocked["status"] = "blocked"
    repo.intents["intent_blocked"] = blocked

    with pytest.raises(PaperServiceError) as exc:
        simulate_order_from_intent_service(
            repo=repo,
            intent_id="intent_blocked",
            execution_config={},
        )
    assert exc.value.status_code == 409


def test_simulate_order_service_idempotent_replay_returns_existing_entities() -> None:
    repo = _FakePaperRepo()
    repo.intents["intent_001"] = _candidate_intent("intent_001")
    repo.executions["intent_001"] = {
        "intent_id": "intent_001",
        "order_id": "order_existing",
        "fill_id": "fill_existing",
    }
    repo.orders["order_existing"] = {
        "order_id": "order_existing",
        "symbol": "BTCUSDT",
        "side": "buy_open_long",
        "quantity": 1.0,
        "status": "filled",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    repo.fills["fill_existing"] = _open_fill("fill_existing")
    repo.snapshots_for_fill["fill_existing"] = {"snapshot_id": "psnap_fill_existing"}

    payload = simulate_order_from_intent_service(
        repo=repo,
        intent_id="intent_001",
        execution_config={},
    )

    _assert_manual_simulation_flags(payload)
    assert payload["order"] == repo.orders["order_existing"]
    assert payload["fill"] == repo.fills["fill_existing"]
    assert payload["portfolio_snapshot"] == repo.snapshots_for_fill["fill_existing"]
    assert repo.lifecycle_events[-1]["event_type"] == "intent_simulation_replayed"
    assert repo.lifecycle_events[-1]["entity_type"] == "paper_intent"
    assert repo.lifecycle_events[-1]["entity_id"] == "intent_001"


def test_simulate_order_service_corrupted_existing_execution_raises_409() -> None:
    repo = _FakePaperRepo()
    repo.intents["intent_001"] = _candidate_intent("intent_001")
    repo.executions["intent_001"] = {
        "intent_id": "intent_001",
        "order_id": "missing_order",
        "fill_id": "missing_fill",
    }

    with pytest.raises(PaperServiceError) as exc:
        simulate_order_from_intent_service(
            repo=repo,
            intent_id="intent_001",
            execution_config={},
        )
    assert exc.value.status_code == 409


def test_close_fill_service_happy_path() -> None:
    repo = _FakePaperRepo()
    repo.fills["fill_001"] = _open_fill("fill_001")

    payload = close_paper_fill_service(
        repo=repo,
        fill_id="fill_001",
        exit_price=105.0,
        exit_fee=0.01,
        close_reason="manual_exit",
        closed_at=datetime(2026, 1, 1, 1, 0, tzinfo=UTC),
    )

    _assert_manual_simulation_flags(payload)
    assert isinstance(payload["close"], dict)
    assert isinstance(payload["portfolio_snapshot"], dict)
    assert payload["close"]["entry_fill_id"] == "fill_001"
    assert repo.lifecycle_events[-1]["event_type"] == "paper_fill_closed"
    assert repo.lifecycle_events[-1]["entity_type"] == "paper_fill"
    assert repo.lifecycle_events[-1]["entity_id"] == "fill_001"


def test_close_fill_service_missing_fill_raises_404() -> None:
    repo = _FakePaperRepo()
    with pytest.raises(PaperServiceError) as exc:
        close_paper_fill_service(
            repo=repo,
            fill_id="missing",
            exit_price=105.0,
            exit_fee=0.0,
            close_reason="manual_exit",
            closed_at=datetime(2026, 1, 1, 1, 0, tzinfo=UTC),
        )
    assert exc.value.status_code == 404


def test_close_fill_service_invalid_close_payload_raises_400() -> None:
    repo = _FakePaperRepo()
    invalid_fill = _open_fill("fill_invalid")
    invalid_fill["quantity"] = 0.0
    repo.fills["fill_invalid"] = invalid_fill

    with pytest.raises(PaperServiceError) as exc:
        close_paper_fill_service(
            repo=repo,
            fill_id="fill_invalid",
            exit_price=105.0,
            exit_fee=0.0,
            close_reason="manual_exit",
            closed_at=datetime(2026, 1, 1, 1, 0, tzinfo=UTC),
        )
    assert exc.value.status_code == 400


def test_close_fill_service_duplicate_close_returns_existing_close_and_snapshot() -> None:
    repo = _FakePaperRepo()
    repo.fills["fill_001"] = _open_fill("fill_001")
    repo.closes_by_fill["fill_001"] = {
        "close_id": "pclose_existing",
        "entry_fill_id": "fill_001",
        "pnl_status": "closed",
    }
    repo.snapshots_for_close["pclose_existing"] = {"snapshot_id": "psnap_existing"}

    payload = close_paper_fill_service(
        repo=repo,
        fill_id="fill_001",
        exit_price=107.0,
        exit_fee=0.0,
        close_reason="manual_exit",
        closed_at=datetime(2026, 1, 1, 1, 0, tzinfo=UTC),
    )

    _assert_manual_simulation_flags(payload)
    assert payload["close"] == repo.closes_by_fill["fill_001"]
    assert payload["portfolio_snapshot"] == repo.snapshots_for_close["pclose_existing"]
    assert repo.lifecycle_events[-1]["event_type"] == "paper_close_replayed"
    assert repo.lifecycle_events[-1]["entity_type"] == "paper_fill"
    assert repo.lifecycle_events[-1]["entity_id"] == "fill_001"
