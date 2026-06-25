from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.api import routes_paper
from app.api.main import app
from app.core.safety import LiveTradingDisabledError
from app.execution.real_disabled import RealBrokerInterface
from app.storage.database import build_engine, build_session_factory, session_scope
from app.storage.repositories import PostgresPaperRepository


def _assert_manual_simulation_flags(payload: dict[str, object]) -> None:
    assert payload["simulation_only"] is True
    assert payload["manual_trigger"] is True
    assert payload["real_execution"] is False


def test_demo_reset_and_paper_endpoints_read_postgres_state() -> None:
    client = TestClient(app)
    reset_response = client.post("/demo/reset")
    assert reset_response.status_code == 200
    assert reset_response.json()["status"] == "reset"

    portfolio_response = client.get("/paper/portfolio")
    assert portfolio_response.status_code == 200
    assert portfolio_response.json()["equity"] == 10_000.0

    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        PostgresPaperRepository(session).create_order_and_fill(
            order_id="paper-order-test",
            fill_id="paper-fill-test",
            symbol="BTCUSDT",
            side="long",
            quantity=0.1,
            price=100.0,
            fee=0.0055,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )

    orders_response = client.get("/paper/orders")
    fills_response = client.get("/paper/fills")
    demo_state_response = client.get("/ui/demo-state")

    assert orders_response.status_code == 200
    assert fills_response.status_code == 200
    assert demo_state_response.status_code == 200
    assert any(
        order["order_id"] == "paper-order-test" for order in orders_response.json()["orders"]
    )
    assert any(fill["fill_id"] == "paper-fill-test" for fill in fills_response.json()["fills"])
    assert any(
        order["order_id"] == "paper-order-test" for order in demo_state_response.json()["orders"]
    )


def test_latest_fills_include_accounting_fields() -> None:
    client = TestClient(app)
    client.post("/demo/reset")

    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        PostgresPaperRepository(session).create_order_and_fill(
            order_id="paper-order-accounting-001",
            fill_id="paper-fill-accounting-001",
            symbol="BTCUSDT",
            side="buy_open_long",
            quantity=0.2,
            price=100.0,
            fee=0.011,
            created_at=datetime(2026, 1, 2, tzinfo=UTC),
        )

    fills_response = client.get("/paper/fills", params={"limit": 10})
    assert fills_response.status_code == 200
    fills = fills_response.json()["fills"]
    fill = next(item for item in fills if item["fill_id"] == "paper-fill-accounting-001")

    assert fill["notional"] == 20.0
    assert fill["fees_paid"] == 0.011
    assert fill["entry_price"] == 100.0
    assert fill["entry_time"] == "2026-01-02T00:00:00+00:00"
    assert fill["pnl_status"] == "open"
    assert fill["realized_pnl"] is None
    assert fill["unrealized_pnl"] is None


def test_paper_approval_endpoint_is_read_only(monkeypatch) -> None:
    def _load_readiness(model_run_id: str, **_: object) -> dict[str, object]:
        return {
            "model_run_id": model_run_id,
            "readiness_status": "research_only",
            "readiness_score": 64,
            "blocking_reasons": ["Strategy comparison data is missing."],
            "warnings": ["Feature importance artifact is unavailable for this model run."],
        }

    monkeypatch.setattr(routes_paper, "load_readiness_report_payload", _load_readiness)

    client = TestClient(app)
    client.post("/demo/reset")

    orders_before = client.get("/paper/orders").json()["orders"]
    fills_before = client.get("/paper/fills").json()["fills"]

    response = client.get("/paper/approval/model_approval_001")
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_run_id"] == "model_approval_001"
    assert payload["paper_demo_status"] == "watch_only"
    assert payload["approved_for_paper_demo"] is False
    assert "approval_score" in payload

    orders_after = client.get("/paper/orders").json()["orders"]
    fills_after = client.get("/paper/fills").json()["fills"]
    assert orders_after == orders_before
    assert fills_after == fills_before


def test_paper_approval_endpoint_returns_blocked_when_readiness_missing(monkeypatch) -> None:
    def _load_readiness(model_run_id: str, **_: object) -> dict[str, object]:
        raise FileNotFoundError(model_run_id)

    monkeypatch.setattr(routes_paper, "load_readiness_report_payload", _load_readiness)

    client = TestClient(app)
    response = client.get("/paper/approval/missing_eval_001")
    assert response.status_code == 200
    payload = response.json()
    assert payload["paper_demo_status"] == "blocked"
    assert payload["approved_for_paper_demo"] is False


def test_paper_intent_endpoint_persists_candidate_and_list_returns_it(monkeypatch) -> None:
    def _load_readiness(model_run_id: str, **_: object) -> dict[str, object]:
        return {
            "model_run_id": model_run_id,
            "readiness_status": "paper_candidate",
            "readiness_score": 90,
            "blocking_reasons": [],
            "warnings": [],
        }

    monkeypatch.setattr(routes_paper, "load_readiness_report_payload", _load_readiness)

    client = TestClient(app)
    client.post("/demo/reset")
    orders_before = client.get("/paper/orders").json()["orders"]
    fills_before = client.get("/paper/fills").json()["fills"]

    response = client.post(
        "/paper/intents/from-signal",
        json={
            "model_run_id": "model_intent_ok_001",
            "signal_payload": {
                "signal_id": "sig_intent_001",
                "symbol": "BTCUSDT",
                "side": "long",
                "event_time": "2026-01-01T00:00:00+00:00",
                "entry_ref": 100.0,
                "stop_ref": 95.0,
                "take_ref": 110.0,
                "ttl_sec": 1800,
                "rationale_json": {"reason": "integration"},
            },
        },
    )
    assert response.status_code == 200
    intent = response.json()
    assert intent["status"] == "candidate"
    assert intent["signal_id"] == "sig_intent_001"

    list_response = client.get("/paper/intents", params={"limit": 10})
    assert list_response.status_code == 200
    intents = list_response.json()["intents"]
    assert any(item["intent_id"] == intent["intent_id"] for item in intents)

    candidate_response = client.get("/paper/intents", params={"status": "candidate"})
    assert candidate_response.status_code == 200
    candidate_intents = candidate_response.json()["intents"]
    assert any(item["intent_id"] == intent["intent_id"] for item in candidate_intents)

    orders_after = client.get("/paper/orders").json()["orders"]
    fills_after = client.get("/paper/fills").json()["fills"]
    assert orders_after == orders_before
    assert fills_after == fills_before


def test_paper_intent_endpoint_blocks_when_readiness_missing(monkeypatch) -> None:
    def _load_readiness(model_run_id: str, **_: object) -> dict[str, object]:
        raise FileNotFoundError(model_run_id)

    monkeypatch.setattr(routes_paper, "load_readiness_report_payload", _load_readiness)

    client = TestClient(app)
    response = client.post(
        "/paper/intents/from-signal",
        json={
            "model_run_id": "missing_eval_001",
            "signal_payload": {
                "signal_id": "sig_intent_missing_001",
                "symbol": "ETHUSDT",
                "side": "short",
                "event_time": "2026-01-01T00:00:00+00:00",
                "entry_ref": 100.0,
                "stop_ref": 105.0,
                "take_ref": 90.0,
                "ttl_sec": 1800,
            },
        },
    )
    assert response.status_code == 200
    intent = response.json()
    assert intent["status"] == "blocked"
    assert intent["approved_for_paper_demo"] is False


def test_paper_intent_endpoint_persists_blocked_invalid_signal(monkeypatch) -> None:
    def _load_readiness(model_run_id: str, **_: object) -> dict[str, object]:
        return {
            "model_run_id": model_run_id,
            "readiness_status": "paper_candidate",
            "readiness_score": 90,
            "blocking_reasons": [],
            "warnings": [],
        }

    monkeypatch.setattr(routes_paper, "load_readiness_report_payload", _load_readiness)

    client = TestClient(app)
    client.post("/demo/reset")
    orders_before = client.get("/paper/orders").json()["orders"]
    fills_before = client.get("/paper/fills").json()["fills"]

    response = client.post(
        "/paper/intents/from-signal",
        json={
            "model_run_id": "model_intent_invalid_001",
            "signal_payload": {
                "signal_id": "sig_intent_invalid_001",
                "symbol": "BTCUSDT",
                "side": "flat",
                "event_time": "2026-01-01T00:00:00+00:00",
                "entry_ref": 100.0,
                "stop_ref": 95.0,
                "take_ref": 110.0,
                "ttl_sec": 1800,
            },
        },
    )
    assert response.status_code == 200
    intent = response.json()
    assert intent["status"] == "blocked"

    blocked_response = client.get("/paper/intents", params={"status": "blocked", "limit": 20})
    assert blocked_response.status_code == 200
    blocked_intents = blocked_response.json()["intents"]
    assert any(item["intent_id"] == intent["intent_id"] for item in blocked_intents)

    orders_after = client.get("/paper/orders").json()["orders"]
    fills_after = client.get("/paper/fills").json()["fills"]
    assert orders_after == orders_before
    assert fills_after == fills_before


def test_simulate_order_rejects_blocked_intent(monkeypatch) -> None:
    def _load_readiness(model_run_id: str, **_: object) -> dict[str, object]:
        raise FileNotFoundError(model_run_id)

    monkeypatch.setattr(routes_paper, "load_readiness_report_payload", _load_readiness)

    client = TestClient(app)
    client.post("/demo/reset")

    intent_response = client.post(
        "/paper/intents/from-signal",
        json={
            "model_run_id": "model_blocked_sim_001",
            "signal_payload": {
                "signal_id": "sig_blocked_sim_001",
                "symbol": "BTCUSDT",
                "side": "long",
                "event_time": "2026-01-01T00:00:00+00:00",
                "entry_ref": 100.0,
                "stop_ref": 95.0,
                "take_ref": 110.0,
                "ttl_sec": 1800,
            },
        },
    )
    intent = intent_response.json()
    assert intent["status"] == "blocked"

    orders_before = client.get("/paper/orders").json()["orders"]
    fills_before = client.get("/paper/fills").json()["fills"]
    simulate_response = client.post(f"/paper/intents/{intent['intent_id']}/simulate-order")
    assert simulate_response.status_code == 409
    detail = simulate_response.json()["detail"]
    assert detail["message"] == "Paper intent is not eligible for manual simulation."
    assert detail["blocking_reasons"]
    assert client.get("/paper/orders").json()["orders"] == orders_before
    assert client.get("/paper/fills").json()["fills"] == fills_before


def test_simulate_order_missing_intent_returns_404() -> None:
    client = TestClient(app)
    client.post("/demo/reset")

    response = client.post("/paper/intents/intent_missing_404/simulate-order")
    assert response.status_code == 404
    assert response.json()["detail"] == "Paper intent not found."


def test_simulate_order_corrupted_existing_execution_returns_409(monkeypatch) -> None:
    def _load_readiness(model_run_id: str, **_: object) -> dict[str, object]:
        return {
            "model_run_id": model_run_id,
            "readiness_status": "paper_candidate",
            "readiness_score": 95,
            "blocking_reasons": [],
            "warnings": [],
        }

    monkeypatch.setattr(routes_paper, "load_readiness_report_payload", _load_readiness)

    client = TestClient(app)
    client.post("/demo/reset")
    intent_response = client.post(
        "/paper/intents/from-signal",
        json={
            "model_run_id": "model_corrupt_exec_001",
            "signal_payload": {
                "signal_id": "sig_corrupt_exec_001",
                "symbol": "BTCUSDT",
                "side": "long",
                "event_time": "2026-01-01T00:00:00+00:00",
                "entry_ref": 100.0,
                "stop_ref": 95.0,
                "take_ref": 110.0,
                "ttl_sec": 1800,
            },
        },
    )
    intent_id = intent_response.json()["intent_id"]

    first_simulate = client.post(f"/paper/intents/{intent_id}/simulate-order")
    assert first_simulate.status_code == 200
    first_payload = first_simulate.json()
    order_id = str(first_payload["order"]["order_id"])

    original_get_paper_order = PostgresPaperRepository.get_paper_order

    def _corrupted_get_paper_order(
        self: PostgresPaperRepository, request_order_id: str
    ) -> dict[str, object] | None:
        if request_order_id == order_id:
            return None
        return original_get_paper_order(self, request_order_id)

    monkeypatch.setattr(PostgresPaperRepository, "get_paper_order", _corrupted_get_paper_order)

    replay_response = client.post(f"/paper/intents/{intent_id}/simulate-order")
    assert replay_response.status_code == 409
    assert (
        replay_response.json()["detail"]
        == "Intent simulation record exists but order/fill is missing."
    )


def test_simulate_order_creates_long_order_fill_and_prevents_duplicates(monkeypatch) -> None:
    def _load_readiness(model_run_id: str, **_: object) -> dict[str, object]:
        return {
            "model_run_id": model_run_id,
            "readiness_status": "paper_candidate",
            "readiness_score": 95,
            "blocking_reasons": [],
            "warnings": [],
        }

    monkeypatch.setattr(routes_paper, "load_readiness_report_payload", _load_readiness)

    client = TestClient(app)
    client.post("/demo/reset")

    intent_response = client.post(
        "/paper/intents/from-signal",
        json={
            "model_run_id": "model_sim_long_001",
            "signal_payload": {
                "signal_id": "sig_sim_long_001",
                "symbol": "BTCUSDT",
                "side": "long",
                "event_time": "2026-01-01T00:00:00+00:00",
                "entry_ref": 100.0,
                "stop_ref": 95.0,
                "take_ref": 110.0,
                "ttl_sec": 1800,
            },
        },
    )
    intent = intent_response.json()
    assert intent["status"] == "candidate"

    simulate_response = client.post(f"/paper/intents/{intent['intent_id']}/simulate-order")
    assert simulate_response.status_code == 200
    payload = simulate_response.json()
    assert {
        "intent_id",
        "source_intent_id",
        "order",
        "fill",
        "portfolio_snapshot",
        "manual_trigger",
        "simulation_only",
        "real_execution",
    }.issubset(payload.keys())
    _assert_manual_simulation_flags(payload)
    assert payload["source_intent_id"] == intent["intent_id"]
    assert payload["order"]["side"] == "buy_open_long"
    assert payload["fill"]["order_id"] == payload["order"]["order_id"]
    assert payload["portfolio_snapshot"]["positions"]["BTCUSDT"]["status"] == "open"
    assert payload["portfolio_snapshot"]["positions"]["BTCUSDT"]["side"] == "long"
    assert (
        payload["portfolio_snapshot"]["positions"]["BTCUSDT"]["entry_fill_id"]
        == payload["fill"]["fill_id"]
    )

    duplicate_response = client.post(f"/paper/intents/{intent['intent_id']}/simulate-order")
    assert duplicate_response.status_code == 200
    duplicate_payload = duplicate_response.json()
    _assert_manual_simulation_flags(duplicate_payload)
    assert duplicate_payload["source_intent_id"] == intent["intent_id"]
    assert duplicate_payload["order"]["order_id"] == payload["order"]["order_id"]
    assert duplicate_payload["fill"]["fill_id"] == payload["fill"]["fill_id"]
    assert (
        duplicate_payload["portfolio_snapshot"]["snapshot_id"]
        == payload["portfolio_snapshot"]["snapshot_id"]
    )

    orders = client.get("/paper/orders").json()["orders"]
    fills = client.get("/paper/fills").json()["fills"]
    portfolio = client.get("/paper/portfolio").json()
    assert len([o for o in orders if o["order_id"] == payload["order"]["order_id"]]) == 1
    assert len([f for f in fills if f["fill_id"] == payload["fill"]["fill_id"]]) == 1
    assert portfolio["positions"]["BTCUSDT"]["status"] == "open"


def test_simulate_order_creates_short_sell_open_short(monkeypatch) -> None:
    def _load_readiness(model_run_id: str, **_: object) -> dict[str, object]:
        return {
            "model_run_id": model_run_id,
            "readiness_status": "paper_candidate",
            "readiness_score": 92,
            "blocking_reasons": [],
            "warnings": [],
        }

    monkeypatch.setattr(routes_paper, "load_readiness_report_payload", _load_readiness)

    client = TestClient(app)
    client.post("/demo/reset")

    intent_response = client.post(
        "/paper/intents/from-signal",
        json={
            "model_run_id": "model_sim_short_001",
            "signal_payload": {
                "signal_id": "sig_sim_short_001",
                "symbol": "ETHUSDT",
                "side": "short",
                "event_time": "2026-01-01T00:00:00+00:00",
                "entry_ref": 100.0,
                "stop_ref": 105.0,
                "take_ref": 90.0,
                "ttl_sec": 1800,
            },
        },
    )
    intent = intent_response.json()
    assert intent["status"] == "candidate"

    simulate_response = client.post(f"/paper/intents/{intent['intent_id']}/simulate-order")
    assert simulate_response.status_code == 200
    payload = simulate_response.json()
    assert payload["order"]["side"] == "sell_open_short"
    assert payload["simulation_only"] is True
    assert payload["portfolio_snapshot"]["positions"]["ETHUSDT"]["status"] == "open"
    assert payload["portfolio_snapshot"]["positions"]["ETHUSDT"]["side"] == "short"


def test_simulate_order_path_does_not_call_real_broker(monkeypatch) -> None:
    def _load_readiness(model_run_id: str, **_: object) -> dict[str, object]:
        return {
            "model_run_id": model_run_id,
            "readiness_status": "paper_candidate",
            "readiness_score": 90,
            "blocking_reasons": [],
            "warnings": [],
        }

    monkeypatch.setattr(routes_paper, "load_readiness_report_payload", _load_readiness)

    def _raise_if_called(*_: object, **__: object) -> None:
        raise AssertionError("Real broker should never be called from paper simulation path.")

    monkeypatch.setattr(RealBrokerInterface, "place_order", _raise_if_called)

    client = TestClient(app)
    client.post("/demo/reset")
    intent_response = client.post(
        "/paper/intents/from-signal",
        json={
            "model_run_id": "model_no_real_001",
            "signal_payload": {
                "signal_id": "sig_no_real_001",
                "symbol": "BTCUSDT",
                "side": "long",
                "event_time": "2026-01-01T00:00:00+00:00",
                "entry_ref": 100.0,
                "stop_ref": 95.0,
                "take_ref": 110.0,
                "ttl_sec": 1800,
            },
        },
    )
    intent_id = intent_response.json()["intent_id"]
    simulate_response = client.post(f"/paper/intents/{intent_id}/simulate-order")
    assert simulate_response.status_code == 200


def test_opposite_side_same_symbol_positions_remain_deterministic(monkeypatch) -> None:
    def _load_readiness(model_run_id: str, **_: object) -> dict[str, object]:
        return {
            "model_run_id": model_run_id,
            "readiness_status": "paper_candidate",
            "readiness_score": 94,
            "blocking_reasons": [],
            "warnings": [],
        }

    monkeypatch.setattr(routes_paper, "load_readiness_report_payload", _load_readiness)

    client = TestClient(app)
    client.post("/demo/reset")

    long_intent = client.post(
        "/paper/intents/from-signal",
        json={
            "model_run_id": "model_pos_long_001",
            "signal_payload": {
                "signal_id": "sig_pos_long_001",
                "symbol": "BTCUSDT",
                "side": "long",
                "event_time": "2026-01-01T00:00:00+00:00",
                "entry_ref": 100.0,
                "stop_ref": 95.0,
                "take_ref": 110.0,
                "ttl_sec": 1800,
            },
        },
    ).json()
    long_sim = client.post(f"/paper/intents/{long_intent['intent_id']}/simulate-order").json()
    long_fill_id = long_sim["fill"]["fill_id"]

    short_intent = client.post(
        "/paper/intents/from-signal",
        json={
            "model_run_id": "model_pos_short_001",
            "signal_payload": {
                "signal_id": "sig_pos_short_001",
                "symbol": "BTCUSDT",
                "side": "short",
                "event_time": "2026-01-01T00:05:00+00:00",
                "entry_ref": 101.0,
                "stop_ref": 106.0,
                "take_ref": 91.0,
                "ttl_sec": 1800,
            },
        },
    ).json()
    short_sim = client.post(f"/paper/intents/{short_intent['intent_id']}/simulate-order")
    assert short_sim.status_code == 200

    portfolio = client.get("/paper/portfolio").json()
    assert "BTCUSDT" in portfolio["positions"]
    assert portfolio["positions"]["BTCUSDT"]["side"] == "long"
    assert portfolio["positions"]["BTCUSDT"]["entry_fill_id"] == long_fill_id


def test_paper_simulation_audit_endpoint_returns_read_only_audit_payload(monkeypatch) -> None:
    def _load_readiness(model_run_id: str, **_: object) -> dict[str, object]:
        return {
            "model_run_id": model_run_id,
            "readiness_status": "paper_candidate",
            "readiness_score": 95,
            "blocking_reasons": [],
            "warnings": [],
        }

    monkeypatch.setattr(routes_paper, "load_readiness_report_payload", _load_readiness)

    client = TestClient(app)
    client.post("/demo/reset")

    candidate_response = client.post(
        "/paper/intents/from-signal",
        json={
            "model_run_id": "model_audit_candidate_001",
            "signal_payload": {
                "signal_id": "sig_audit_candidate_001",
                "symbol": "BTCUSDT",
                "side": "long",
                "event_time": "2026-01-01T00:00:00+00:00",
                "entry_ref": 100.0,
                "stop_ref": 95.0,
                "take_ref": 110.0,
                "ttl_sec": 1800,
            },
        },
    )
    assert candidate_response.status_code == 200
    candidate_intent = candidate_response.json()
    assert candidate_intent["status"] == "candidate"

    blocked_response = client.post(
        "/paper/intents/from-signal",
        json={
            "model_run_id": "model_audit_blocked_001",
            "signal_payload": {
                "signal_id": "sig_audit_blocked_001",
                "symbol": "ETHUSDT",
                "side": "flat",
                "event_time": "2026-01-01T00:00:00+00:00",
                "entry_ref": 100.0,
                "stop_ref": 95.0,
                "take_ref": 110.0,
                "ttl_sec": 1800,
            },
        },
    )
    assert blocked_response.status_code == 200
    assert blocked_response.json()["status"] == "blocked"

    simulate_response = client.post(
        f"/paper/intents/{candidate_intent['intent_id']}/simulate-order"
    )
    assert simulate_response.status_code == 200

    audit_response = client.get("/paper/simulation-audit", params={"limit": 50})
    assert audit_response.status_code == 200
    payload = audit_response.json()

    assert "created_at" in payload
    assert "summary" in payload
    assert "intent_status_counts" in payload
    assert "simulation_status_counts" in payload
    assert "blocking_reason_counts" in payload
    assert "latest_intents" in payload
    assert "latest_simulations" in payload
    assert "latest_orders" in payload
    assert "latest_fills" in payload
    assert "portfolio_summary" in payload
    assert "warnings" in payload

    assert payload["summary"]["total_intents"] >= 2
    assert payload["summary"]["blocked_intents"] >= 1
    assert payload["summary"]["candidate_intents"] >= 1
    assert payload["summary"]["simulated_intents"] >= 1
    assert payload["summary"]["unsimulated_candidate_intents"] >= 0


def test_paper_simulation_audit_endpoint_does_not_mutate_state(monkeypatch) -> None:
    def _load_readiness(model_run_id: str, **_: object) -> dict[str, object]:
        return {
            "model_run_id": model_run_id,
            "readiness_status": "paper_candidate",
            "readiness_score": 93,
            "blocking_reasons": [],
            "warnings": [],
        }

    monkeypatch.setattr(routes_paper, "load_readiness_report_payload", _load_readiness)

    client = TestClient(app)
    client.post("/demo/reset")
    client.post(
        "/paper/intents/from-signal",
        json={
            "model_run_id": "model_audit_nomutate_001",
            "signal_payload": {
                "signal_id": "sig_audit_nomutate_001",
                "symbol": "BTCUSDT",
                "side": "long",
                "event_time": "2026-01-01T00:00:00+00:00",
                "entry_ref": 100.0,
                "stop_ref": 95.0,
                "take_ref": 110.0,
                "ttl_sec": 1800,
            },
        },
    )

    intents_before = client.get("/paper/intents", params={"limit": 200}).json()["intents"]
    orders_before = client.get("/paper/orders", params={"limit": 200}).json()["orders"]
    fills_before = client.get("/paper/fills", params={"limit": 200}).json()["fills"]

    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        executions_before = PostgresPaperRepository(session).latest_paper_intent_executions(
            limit=200
        )

    response = client.get("/paper/simulation-audit", params={"limit": 200})
    assert response.status_code == 200

    intents_after = client.get("/paper/intents", params={"limit": 200}).json()["intents"]
    orders_after = client.get("/paper/orders", params={"limit": 200}).json()["orders"]
    fills_after = client.get("/paper/fills", params={"limit": 200}).json()["fills"]
    with session_scope(session_factory) as session:
        executions_after = PostgresPaperRepository(session).latest_paper_intent_executions(
            limit=200
        )

    assert intents_after == intents_before
    assert orders_after == orders_before
    assert fills_after == fills_before
    assert executions_after == executions_before


def test_paper_track_record_endpoint_returns_read_only_payload() -> None:
    client = TestClient(app)
    client.post("/demo/reset")

    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        repo = PostgresPaperRepository(session)
        repo.create_order_and_fill(
            order_id="paper-track-order-001",
            fill_id="paper-track-fill-001",
            symbol="BTCUSDT",
            side="buy_open_long",
            quantity=0.5,
            price=100.0,
            fee=0.0275,
            created_at=datetime(2026, 1, 3, tzinfo=UTC),
        )

    response = client.get("/paper/track-record", params={"limit": 500})
    assert response.status_code == 200
    payload = response.json()

    assert "created_at" in payload
    assert "summary" in payload
    assert "performance_metrics" in payload
    assert "risk_metrics" in payload
    assert "daily_summary" in payload
    assert "monthly_summary" in payload
    assert "latest_trades" in payload
    assert "equity_curve" in payload
    assert "warnings" in payload
    assert payload["summary"]["total_orders"] >= 1
    assert payload["summary"]["total_fills"] >= 1


def test_paper_track_record_endpoint_does_not_mutate_state() -> None:
    client = TestClient(app)
    client.post("/demo/reset")

    intents_before = client.get("/paper/intents", params={"limit": 200}).json()["intents"]
    orders_before = client.get("/paper/orders", params={"limit": 200}).json()["orders"]
    fills_before = client.get("/paper/fills", params={"limit": 200}).json()["fills"]

    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        executions_before = PostgresPaperRepository(session).latest_paper_intent_executions(
            limit=200
        )
        snapshots_before = PostgresPaperRepository(session).latest_portfolio_snapshots(limit=200)

    response = client.get("/paper/track-record", params={"limit": 500})
    assert response.status_code == 200

    intents_after = client.get("/paper/intents", params={"limit": 200}).json()["intents"]
    orders_after = client.get("/paper/orders", params={"limit": 200}).json()["orders"]
    fills_after = client.get("/paper/fills", params={"limit": 200}).json()["fills"]
    with session_scope(session_factory) as session:
        executions_after = PostgresPaperRepository(session).latest_paper_intent_executions(
            limit=200
        )
        snapshots_after = PostgresPaperRepository(session).latest_portfolio_snapshots(limit=200)

    assert intents_after == intents_before
    assert orders_after == orders_before
    assert fills_after == fills_before
    assert executions_after == executions_before
    assert snapshots_after == snapshots_before


def test_paper_portfolio_audit_endpoint_returns_read_only_payload(monkeypatch) -> None:
    def _load_readiness(model_run_id: str, **_: object) -> dict[str, object]:
        return {
            "model_run_id": model_run_id,
            "readiness_status": "paper_candidate",
            "readiness_score": 95,
            "blocking_reasons": [],
            "warnings": [],
        }

    monkeypatch.setattr(routes_paper, "load_readiness_report_payload", _load_readiness)

    client = TestClient(app)
    client.post("/demo/reset")

    intent_response = client.post(
        "/paper/intents/from-signal",
        json={
            "model_run_id": "model_portfolio_audit_001",
            "signal_payload": {
                "signal_id": "sig_portfolio_audit_001",
                "symbol": "BTCUSDT",
                "side": "long",
                "event_time": "2026-01-01T00:00:00+00:00",
                "entry_ref": 100.0,
                "stop_ref": 95.0,
                "take_ref": 110.0,
                "ttl_sec": 1800,
            },
        },
    )
    intent_id = intent_response.json()["intent_id"]
    simulate_response = client.post(f"/paper/intents/{intent_id}/simulate-order")
    fill_id = simulate_response.json()["fill"]["fill_id"]
    close_response = client.post(
        f"/paper/fills/{fill_id}/close",
        json={"exit_price": 105.0, "exit_fee": 0.01},
    )
    assert close_response.status_code == 200

    response = client.get("/paper/portfolio-audit", params={"limit": 500})
    assert response.status_code == 200
    payload = response.json()

    assert "created_at" in payload
    assert "summary" in payload
    assert "open_positions" in payload
    assert "closed_trade_count" in payload
    assert "open_fill_count" in payload
    assert "orphan_fills" in payload
    assert "orphan_closes" in payload
    assert "position_consistency_warnings" in payload
    assert "equity_consistency" in payload
    assert "warnings" in payload


def test_paper_portfolio_audit_endpoint_does_not_mutate_state() -> None:
    client = TestClient(app)
    client.post("/demo/reset")

    intents_before = client.get("/paper/intents", params={"limit": 200}).json()["intents"]
    orders_before = client.get("/paper/orders", params={"limit": 200}).json()["orders"]
    fills_before = client.get("/paper/fills", params={"limit": 200}).json()["fills"]

    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        repo = PostgresPaperRepository(session)
        executions_before = repo.latest_paper_intent_executions(limit=200)
        closes_before = repo.latest_paper_trade_closes(limit=200)
        snapshots_before = repo.latest_portfolio_snapshots(limit=200)

    response = client.get("/paper/portfolio-audit", params={"limit": 500})
    assert response.status_code == 200

    intents_after = client.get("/paper/intents", params={"limit": 200}).json()["intents"]
    orders_after = client.get("/paper/orders", params={"limit": 200}).json()["orders"]
    fills_after = client.get("/paper/fills", params={"limit": 200}).json()["fills"]
    with session_scope(session_factory) as session:
        repo = PostgresPaperRepository(session)
        executions_after = repo.latest_paper_intent_executions(limit=200)
        closes_after = repo.latest_paper_trade_closes(limit=200)
        snapshots_after = repo.latest_portfolio_snapshots(limit=200)

    assert intents_after == intents_before
    assert orders_after == orders_before
    assert fills_after == fills_before
    assert executions_after == executions_before
    assert closes_after == closes_before
    assert snapshots_after == snapshots_before


def test_close_fill_endpoint_closes_open_fill_and_is_idempotent(monkeypatch) -> None:
    def _load_readiness(model_run_id: str, **_: object) -> dict[str, object]:
        return {
            "model_run_id": model_run_id,
            "readiness_status": "paper_candidate",
            "readiness_score": 95,
            "blocking_reasons": [],
            "warnings": [],
        }

    monkeypatch.setattr(routes_paper, "load_readiness_report_payload", _load_readiness)

    client = TestClient(app)
    client.post("/demo/reset")

    intent_response = client.post(
        "/paper/intents/from-signal",
        json={
            "model_run_id": "model_close_manual_001",
            "signal_payload": {
                "signal_id": "sig_close_manual_001",
                "symbol": "BTCUSDT",
                "side": "long",
                "event_time": "2026-01-01T00:00:00+00:00",
                "entry_ref": 100.0,
                "stop_ref": 95.0,
                "take_ref": 110.0,
                "ttl_sec": 1800,
            },
        },
    )
    intent_id = intent_response.json()["intent_id"]
    simulate_response = client.post(f"/paper/intents/{intent_id}/simulate-order")
    assert simulate_response.status_code == 200
    fill_id = simulate_response.json()["fill"]["fill_id"]
    portfolio_before = client.get("/paper/portfolio").json()

    close_response = client.post(
        f"/paper/fills/{fill_id}/close",
        json={
            "exit_price": 108.0,
            "exit_fee": 0.05,
            "close_reason": "manual_exit",
        },
    )
    assert close_response.status_code == 200
    close_payload = close_response.json()
    assert {
        "simulation_only",
        "manual_trigger",
        "real_execution",
        "close",
        "portfolio_snapshot",
    }.issubset(close_payload.keys())
    _assert_manual_simulation_flags(close_payload)
    assert close_payload["close"]["entry_fill_id"] == fill_id
    assert close_payload["close"]["pnl_status"] == "closed"
    assert close_payload["close"]["realized_pnl"] > 0.0
    assert "portfolio_snapshot" in close_payload
    realized = float(close_payload["close"]["realized_pnl"])
    snapshot = close_payload["portfolio_snapshot"]
    assert snapshot["equity"] == pytest.approx(float(portfolio_before["equity"]) + realized)
    assert snapshot["available_margin"] == pytest.approx(
        float(portfolio_before["available_margin"]) + realized
    )
    assert snapshot["margin_used"] == pytest.approx(float(portfolio_before["margin_used"]))
    assert "BTCUSDT" not in snapshot["positions"]

    duplicate_close = client.post(
        f"/paper/fills/{fill_id}/close",
        json={"exit_price": 107.0, "exit_fee": 0.0},
    )
    assert duplicate_close.status_code == 200
    duplicate_payload = duplicate_close.json()
    _assert_manual_simulation_flags(duplicate_payload)
    assert duplicate_payload["close"]["close_id"] == close_payload["close"]["close_id"]
    assert duplicate_payload["portfolio_snapshot"]["snapshot_id"] == snapshot["snapshot_id"]
    assert "BTCUSDT" not in duplicate_payload["portfolio_snapshot"]["positions"]

    fills_response = client.get("/paper/fills", params={"limit": 50})
    assert fills_response.status_code == 200
    closed_fill = next(
        item for item in fills_response.json()["fills"] if item["fill_id"] == fill_id
    )
    assert closed_fill["pnl_status"] == "closed"
    assert closed_fill["realized_pnl"] == close_payload["close"]["realized_pnl"]
    assert closed_fill["exit_price"] == 108.0
    assert closed_fill["close_reason"] == "manual_exit"

    portfolio_after_duplicate = client.get("/paper/portfolio").json()
    assert portfolio_after_duplicate["snapshot_id"] == snapshot["snapshot_id"]
    assert portfolio_after_duplicate["equity"] == pytest.approx(snapshot["equity"])
    assert "BTCUSDT" not in portfolio_after_duplicate["positions"]


def test_close_fill_missing_fill_returns_404() -> None:
    client = TestClient(app)
    client.post("/demo/reset")

    response = client.post("/paper/fills/fill_missing_404/close", json={"exit_price": 105.0})
    assert response.status_code == 404
    assert response.json()["detail"] == "Paper fill not found."


def test_close_fill_invalid_payload_returns_400(monkeypatch) -> None:
    def _load_readiness(model_run_id: str, **_: object) -> dict[str, object]:
        return {
            "model_run_id": model_run_id,
            "readiness_status": "paper_candidate",
            "readiness_score": 95,
            "blocking_reasons": [],
            "warnings": [],
        }

    monkeypatch.setattr(routes_paper, "load_readiness_report_payload", _load_readiness)

    client = TestClient(app)
    client.post("/demo/reset")
    intent_response = client.post(
        "/paper/intents/from-signal",
        json={
            "model_run_id": "model_close_invalid_001",
            "signal_payload": {
                "signal_id": "sig_close_invalid_001",
                "symbol": "BTCUSDT",
                "side": "long",
                "event_time": "2026-01-01T00:00:00+00:00",
                "entry_ref": 100.0,
                "stop_ref": 95.0,
                "take_ref": 110.0,
                "ttl_sec": 1800,
            },
        },
    )
    fill_id = client.post(
        f"/paper/intents/{intent_response.json()['intent_id']}/simulate-order"
    ).json()["fill"]["fill_id"]

    response = client.post(
        f"/paper/fills/{fill_id}/close",
        json={"exit_price": 0.0, "exit_fee": 0.0},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "exit_price must be positive"


def test_exit_preview_returns_expected_outcome_and_flags(monkeypatch) -> None:
    def _load_readiness(model_run_id: str, **_: object) -> dict[str, object]:
        return {
            "model_run_id": model_run_id,
            "readiness_status": "paper_candidate",
            "readiness_score": 95,
            "blocking_reasons": [],
            "warnings": [],
        }

    monkeypatch.setattr(routes_paper, "load_readiness_report_payload", _load_readiness)

    client = TestClient(app)
    client.post("/demo/reset")
    intent_response = client.post(
        "/paper/intents/from-signal",
        json={
            "model_run_id": "model_exit_preview_ok_001",
            "signal_payload": {
                "signal_id": "sig_exit_preview_ok_001",
                "symbol": "BTCUSDT",
                "side": "long",
                "event_time": "2026-01-01T00:00:00+00:00",
                "entry_ref": 100.0,
                "stop_ref": 95.0,
                "take_ref": 110.0,
                "ttl_sec": 600,
            },
        },
    )
    fill_id = client.post(
        f"/paper/intents/{intent_response.json()['intent_id']}/simulate-order"
    ).json()["fill"]["fill_id"]
    fill_entry_time = next(
        item["filled_at"]
        for item in client.get("/paper/fills", params={"limit": 200}).json()["fills"]
        if item["fill_id"] == fill_id
    )
    entry_dt = datetime.fromisoformat(fill_entry_time.replace("Z", "+00:00"))
    bar_dt = entry_dt + timedelta(minutes=5)

    response = client.post(
        f"/paper/fills/{fill_id}/exit-preview",
        json={
            "signal": {
                "side": "long",
                "entry_ref": 100.0,
                "stop_ref": 95.0,
                "take_ref": 110.0,
                "ttl_sec": 600,
                "event_time": fill_entry_time,
            },
            "ohlc_bars": [
                {
                    "event_time": bar_dt.isoformat(),
                    "open": 100.0,
                    "high": 111.0,
                    "low": 99.0,
                    "close": 105.0,
                }
            ],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["fill_id"] == fill_id
    assert payload["preview_only"] is True
    assert payload["simulation_only"] is True
    assert payload["manual_trigger"] is True
    assert payload["real_execution"] is False
    assert payload["exit_preview"]["exit_reason"] == "take_profit"
    assert payload["exit_preview"]["exit_price"] == 110.0
    assert payload["exit_preview"]["source"] == "historical_bars_exit_policy"
    assert payload["exit_preview"]["simulation_only"] is True


def test_exit_preview_missing_fill_returns_404() -> None:
    client = TestClient(app)
    client.post("/demo/reset")

    response = client.post(
        "/paper/fills/fill_missing_preview_404/exit-preview",
        json={"signal": {"side": "long"}, "ohlc_bars": []},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Paper fill not found."


@pytest.mark.parametrize(
    ("request_payload", "expected_detail"),
    [
        (
            {"signal": [], "ohlc_bars": []},
            "signal is required and must be an object.",
        ),
        (
            {"signal": {"side": "long"}, "ohlc_bars": {}},
            "ohlc_bars is required and must be a list of objects.",
        ),
        (
            {"signal": {"side": "long"}, "ohlc_bars": ["not-a-bar"]},
            "ohlc_bars must be a list of objects.",
        ),
    ],
)
def test_exit_preview_invalid_payload_shape_returns_400(
    request_payload: dict[str, object],
    expected_detail: str,
) -> None:
    client = TestClient(app)

    response = client.post(
        "/paper/fills/fill_payload_validation_short_circuit/exit-preview",
        json=request_payload,
    )
    assert response.status_code == 400
    assert response.json()["detail"] == expected_detail


def test_exit_preview_invalid_geometry_returns_400(monkeypatch) -> None:
    def _load_readiness(model_run_id: str, **_: object) -> dict[str, object]:
        return {
            "model_run_id": model_run_id,
            "readiness_status": "paper_candidate",
            "readiness_score": 95,
            "blocking_reasons": [],
            "warnings": [],
        }

    monkeypatch.setattr(routes_paper, "load_readiness_report_payload", _load_readiness)

    client = TestClient(app)
    client.post("/demo/reset")
    intent_response = client.post(
        "/paper/intents/from-signal",
        json={
            "model_run_id": "model_exit_preview_bad_geom_001",
            "signal_payload": {
                "signal_id": "sig_exit_preview_bad_geom_001",
                "symbol": "BTCUSDT",
                "side": "long",
                "event_time": "2026-01-01T00:00:00+00:00",
                "entry_ref": 100.0,
                "stop_ref": 95.0,
                "take_ref": 110.0,
                "ttl_sec": 600,
            },
        },
    )
    fill_id = client.post(
        f"/paper/intents/{intent_response.json()['intent_id']}/simulate-order"
    ).json()["fill"]["fill_id"]

    response = client.post(
        f"/paper/fills/{fill_id}/exit-preview",
        json={
            "signal": {
                "side": "long",
                "entry_ref": 100.0,
                "stop_ref": 120.0,
                "take_ref": 110.0,
                "ttl_sec": 600,
                "event_time": "2026-01-01T00:00:00+00:00",
            },
            "ohlc_bars": [],
        },
    )
    assert response.status_code == 400
    assert "Invalid long geometry" in response.json()["detail"]


def test_exit_preview_endpoint_is_read_only(monkeypatch) -> None:
    def _load_readiness(model_run_id: str, **_: object) -> dict[str, object]:
        return {
            "model_run_id": model_run_id,
            "readiness_status": "paper_candidate",
            "readiness_score": 95,
            "blocking_reasons": [],
            "warnings": [],
        }

    monkeypatch.setattr(routes_paper, "load_readiness_report_payload", _load_readiness)

    client = TestClient(app)
    client.post("/demo/reset")
    intent_response = client.post(
        "/paper/intents/from-signal",
        json={
            "model_run_id": "model_exit_preview_nomutate_001",
            "signal_payload": {
                "signal_id": "sig_exit_preview_nomutate_001",
                "symbol": "BTCUSDT",
                "side": "long",
                "event_time": "2026-01-01T00:00:00+00:00",
                "entry_ref": 100.0,
                "stop_ref": 95.0,
                "take_ref": 110.0,
                "ttl_sec": 600,
            },
        },
    )
    fill_id = client.post(
        f"/paper/intents/{intent_response.json()['intent_id']}/simulate-order"
    ).json()["fill"]["fill_id"]

    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        repo = PostgresPaperRepository(session)
        closes_before = repo.latest_paper_trade_closes(limit=200)
        snapshots_before = repo.latest_portfolio_snapshots(limit=200)
        lifecycle_before = repo.latest_paper_lifecycle_events(limit=200)

    response = client.post(
        f"/paper/fills/{fill_id}/exit-preview",
        json={
            "signal": {
                "side": "long",
                "entry_ref": 100.0,
                "stop_ref": 95.0,
                "take_ref": 110.0,
                "ttl_sec": 600,
                "event_time": "2026-01-01T00:00:00+00:00",
            },
            "ohlc_bars": [
                {
                    "event_time": "2026-01-01T00:05:00+00:00",
                    "open": 100.0,
                    "high": 106.0,
                    "low": 98.0,
                    "close": 103.0,
                }
            ],
        },
    )
    assert response.status_code == 200

    with session_scope(session_factory) as session:
        repo = PostgresPaperRepository(session)
        closes_after = repo.latest_paper_trade_closes(limit=200)
        snapshots_after = repo.latest_portfolio_snapshots(limit=200)
        lifecycle_after = repo.latest_paper_lifecycle_events(limit=200)

    assert closes_after == closes_before
    assert snapshots_after == snapshots_before
    assert lifecycle_after == lifecycle_before


def test_exit_preview_keeps_manual_close_behavior_unchanged(monkeypatch) -> None:
    def _load_readiness(model_run_id: str, **_: object) -> dict[str, object]:
        return {
            "model_run_id": model_run_id,
            "readiness_status": "paper_candidate",
            "readiness_score": 95,
            "blocking_reasons": [],
            "warnings": [],
        }

    monkeypatch.setattr(routes_paper, "load_readiness_report_payload", _load_readiness)

    client = TestClient(app)
    client.post("/demo/reset")
    intent_response = client.post(
        "/paper/intents/from-signal",
        json={
            "model_run_id": "model_exit_preview_close_001",
            "signal_payload": {
                "signal_id": "sig_exit_preview_close_001",
                "symbol": "BTCUSDT",
                "side": "long",
                "event_time": "2026-01-01T00:00:00+00:00",
                "entry_ref": 100.0,
                "stop_ref": 95.0,
                "take_ref": 110.0,
                "ttl_sec": 600,
            },
        },
    )
    fill_id = client.post(
        f"/paper/intents/{intent_response.json()['intent_id']}/simulate-order"
    ).json()["fill"]["fill_id"]

    preview_response = client.post(
        f"/paper/fills/{fill_id}/exit-preview",
        json={
            "signal": {
                "side": "long",
                "entry_ref": 100.0,
                "stop_ref": 95.0,
                "take_ref": 110.0,
                "ttl_sec": 600,
                "event_time": "2026-01-01T00:00:00+00:00",
            },
            "ohlc_bars": [],
        },
    )
    assert preview_response.status_code == 200

    close_response = client.post(
        f"/paper/fills/{fill_id}/close",
        json={"exit_price": 108.0, "exit_fee": 0.05},
    )
    assert close_response.status_code == 200
    close_payload = close_response.json()
    _assert_manual_simulation_flags(close_payload)
    assert close_payload["close"]["entry_fill_id"] == fill_id
    assert close_payload["close"]["pnl_status"] == "closed"


def test_close_fill_enables_track_record_closed_trade_metrics(monkeypatch) -> None:
    def _load_readiness(model_run_id: str, **_: object) -> dict[str, object]:
        return {
            "model_run_id": model_run_id,
            "readiness_status": "paper_candidate",
            "readiness_score": 95,
            "blocking_reasons": [],
            "warnings": [],
        }

    monkeypatch.setattr(routes_paper, "load_readiness_report_payload", _load_readiness)

    client = TestClient(app)
    client.post("/demo/reset")

    intent_response = client.post(
        "/paper/intents/from-signal",
        json={
            "model_run_id": "model_close_track_001",
            "signal_payload": {
                "signal_id": "sig_close_track_001",
                "symbol": "ETHUSDT",
                "side": "short",
                "event_time": "2026-01-01T00:00:00+00:00",
                "entry_ref": 100.0,
                "stop_ref": 105.0,
                "take_ref": 90.0,
                "ttl_sec": 1800,
            },
        },
    )
    intent_id = intent_response.json()["intent_id"]
    simulate_response = client.post(f"/paper/intents/{intent_id}/simulate-order")
    fill_id = simulate_response.json()["fill"]["fill_id"]

    close_response = client.post(
        f"/paper/fills/{fill_id}/close",
        json={
            "exit_price": 95.0,
            "exit_fee": 0.02,
        },
    )
    assert close_response.status_code == 200

    track_response = client.get("/paper/track-record", params={"limit": 500})
    assert track_response.status_code == 200
    report = track_response.json()
    summary = report["summary"]
    metrics = report["performance_metrics"]
    assert summary["closed_trades"] >= 1
    assert summary["open_trades"] >= 0
    assert metrics["win_count"] is not None
    assert metrics["total_pnl"] is not None
    assert len(report["equity_curve"]) >= 2
    assert not any(
        "Portfolio equity is not automatically updated by close simulation yet." in warning
        for warning in report["warnings"]
    )
    assert "ETHUSDT" not in client.get("/paper/portfolio").json()["positions"]


def test_close_fill_with_negative_realized_pnl_reduces_portfolio_equity(monkeypatch) -> None:
    def _load_readiness(model_run_id: str, **_: object) -> dict[str, object]:
        return {
            "model_run_id": model_run_id,
            "readiness_status": "paper_candidate",
            "readiness_score": 95,
            "blocking_reasons": [],
            "warnings": [],
        }

    monkeypatch.setattr(routes_paper, "load_readiness_report_payload", _load_readiness)

    client = TestClient(app)
    client.post("/demo/reset")

    intent_response = client.post(
        "/paper/intents/from-signal",
        json={
            "model_run_id": "model_close_loss_001",
            "signal_payload": {
                "signal_id": "sig_close_loss_001",
                "symbol": "BTCUSDT",
                "side": "long",
                "event_time": "2026-01-01T00:00:00+00:00",
                "entry_ref": 100.0,
                "stop_ref": 95.0,
                "take_ref": 110.0,
                "ttl_sec": 1800,
            },
        },
    )
    intent_id = intent_response.json()["intent_id"]
    simulate_response = client.post(f"/paper/intents/{intent_id}/simulate-order")
    fill_id = simulate_response.json()["fill"]["fill_id"]
    portfolio_before = client.get("/paper/portfolio").json()

    close_response = client.post(
        f"/paper/fills/{fill_id}/close",
        json={"exit_price": 95.0, "exit_fee": 0.01},
    )
    assert close_response.status_code == 200
    payload = close_response.json()
    realized = float(payload["close"]["realized_pnl"])
    assert realized < 0.0

    portfolio_after = client.get("/paper/portfolio").json()
    assert portfolio_after["equity"] == pytest.approx(float(portfolio_before["equity"]) + realized)
    assert portfolio_after["available_margin"] == pytest.approx(
        float(portfolio_before["available_margin"]) + realized
    )


def test_close_fill_path_does_not_call_real_broker(monkeypatch) -> None:
    def _load_readiness(model_run_id: str, **_: object) -> dict[str, object]:
        return {
            "model_run_id": model_run_id,
            "readiness_status": "paper_candidate",
            "readiness_score": 95,
            "blocking_reasons": [],
            "warnings": [],
        }

    monkeypatch.setattr(routes_paper, "load_readiness_report_payload", _load_readiness)

    def _raise_if_called(*_: object, **__: object) -> None:
        raise AssertionError("Real broker should never be called from paper close path.")

    monkeypatch.setattr(RealBrokerInterface, "place_order", _raise_if_called)

    client = TestClient(app)
    client.post("/demo/reset")
    intent_response = client.post(
        "/paper/intents/from-signal",
        json={
            "model_run_id": "model_close_no_real_001",
            "signal_payload": {
                "signal_id": "sig_close_no_real_001",
                "symbol": "BTCUSDT",
                "side": "long",
                "event_time": "2026-01-01T00:00:00+00:00",
                "entry_ref": 100.0,
                "stop_ref": 95.0,
                "take_ref": 110.0,
                "ttl_sec": 1800,
            },
        },
    )
    intent_id = intent_response.json()["intent_id"]
    simulate_response = client.post(f"/paper/intents/{intent_id}/simulate-order")
    fill_id = simulate_response.json()["fill"]["fill_id"]
    close_response = client.post(
        f"/paper/fills/{fill_id}/close",
        json={"exit_price": 105.0, "exit_fee": 0.01},
    )
    assert close_response.status_code == 200


def test_paper_lifecycle_events_logged_and_filterable(monkeypatch) -> None:
    def _load_readiness(model_run_id: str, **_: object) -> dict[str, object]:
        return {
            "model_run_id": model_run_id,
            "readiness_status": "paper_candidate",
            "readiness_score": 95,
            "blocking_reasons": [],
            "warnings": [],
        }

    monkeypatch.setattr(routes_paper, "load_readiness_report_payload", _load_readiness)

    client = TestClient(app)
    client.post("/demo/reset")

    intent_response = client.post(
        "/paper/intents/from-signal",
        json={
            "model_run_id": "model_lifecycle_001",
            "signal_payload": {
                "signal_id": "sig_lifecycle_001",
                "symbol": "BTCUSDT",
                "side": "long",
                "event_time": "2026-01-01T00:00:00+00:00",
                "entry_ref": 100.0,
                "stop_ref": 95.0,
                "take_ref": 110.0,
                "ttl_sec": 1800,
            },
        },
    )
    assert intent_response.status_code == 200
    intent_id = intent_response.json()["intent_id"]

    simulate_first = client.post(f"/paper/intents/{intent_id}/simulate-order")
    assert simulate_first.status_code == 200
    fill_id = simulate_first.json()["fill"]["fill_id"]

    simulate_replay = client.post(f"/paper/intents/{intent_id}/simulate-order")
    assert simulate_replay.status_code == 200

    close_first = client.post(
        f"/paper/fills/{fill_id}/close",
        json={"exit_price": 107.0, "exit_fee": 0.01},
    )
    assert close_first.status_code == 200

    close_replay = client.post(
        f"/paper/fills/{fill_id}/close",
        json={"exit_price": 106.0, "exit_fee": 0.0},
    )
    assert close_replay.status_code == 200

    events_response = client.get("/paper/lifecycle-events", params={"limit": 500})
    assert events_response.status_code == 200
    events_payload = events_response.json()
    events = events_payload["events"]
    assert events_payload["count"] == len(events)

    event_types = {event["event_type"] for event in events}
    assert "intent_created" in event_types
    assert "intent_simulated" in event_types
    assert "intent_simulation_replayed" in event_types
    assert "paper_fill_closed" in event_types
    assert "paper_close_replayed" in event_types

    intent_created_events = client.get(
        "/paper/lifecycle-events",
        params={"event_type": "intent_created", "limit": 100},
    ).json()["events"]
    matched_created = [item for item in intent_created_events if item["entity_id"] == intent_id]
    assert len(matched_created) == 1

    intent_events = client.get(
        "/paper/lifecycle-events",
        params={
            "entity_type": "paper_intent",
            "entity_id": intent_id,
            "limit": 100,
        },
    ).json()["events"]
    intent_event_types = {item["event_type"] for item in intent_events}
    assert "intent_created" in intent_event_types
    assert "intent_simulated" in intent_event_types
    assert "intent_simulation_replayed" in intent_event_types


def test_paper_lifecycle_events_endpoint_is_read_only(monkeypatch) -> None:
    def _load_readiness(model_run_id: str, **_: object) -> dict[str, object]:
        return {
            "model_run_id": model_run_id,
            "readiness_status": "paper_candidate",
            "readiness_score": 95,
            "blocking_reasons": [],
            "warnings": [],
        }

    monkeypatch.setattr(routes_paper, "load_readiness_report_payload", _load_readiness)

    client = TestClient(app)
    client.post("/demo/reset")
    client.post(
        "/paper/intents/from-signal",
        json={
            "model_run_id": "model_lifecycle_nomutate_001",
            "signal_payload": {
                "signal_id": "sig_lifecycle_nomutate_001",
                "symbol": "BTCUSDT",
                "side": "long",
                "event_time": "2026-01-01T00:00:00+00:00",
                "entry_ref": 100.0,
                "stop_ref": 95.0,
                "take_ref": 110.0,
                "ttl_sec": 1800,
            },
        },
    )

    intents_before = client.get("/paper/intents", params={"limit": 200}).json()["intents"]
    orders_before = client.get("/paper/orders", params={"limit": 200}).json()["orders"]
    fills_before = client.get("/paper/fills", params={"limit": 200}).json()["fills"]

    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        repo = PostgresPaperRepository(session)
        executions_before = repo.latest_paper_intent_executions(limit=200)
        closes_before = repo.latest_paper_trade_closes(limit=200)
        snapshots_before = repo.latest_portfolio_snapshots(limit=200)
        lifecycle_before = repo.latest_paper_lifecycle_events(limit=200)

    response = client.get("/paper/lifecycle-events", params={"limit": 200})
    assert response.status_code == 200

    intents_after = client.get("/paper/intents", params={"limit": 200}).json()["intents"]
    orders_after = client.get("/paper/orders", params={"limit": 200}).json()["orders"]
    fills_after = client.get("/paper/fills", params={"limit": 200}).json()["fills"]
    with session_scope(session_factory) as session:
        repo = PostgresPaperRepository(session)
        executions_after = repo.latest_paper_intent_executions(limit=200)
        closes_after = repo.latest_paper_trade_closes(limit=200)
        snapshots_after = repo.latest_portfolio_snapshots(limit=200)
        lifecycle_after = repo.latest_paper_lifecycle_events(limit=200)

    assert intents_after == intents_before
    assert orders_after == orders_before
    assert fills_after == fills_before
    assert executions_after == executions_before
    assert closes_after == closes_before
    assert snapshots_after == snapshots_before
    assert lifecycle_after == lifecycle_before


def test_live_execution_remains_disabled() -> None:
    with pytest.raises(LiveTradingDisabledError):
        RealBrokerInterface().place_order({"symbol": "BTCUSDT", "side": "Buy", "qty": 0.001})
