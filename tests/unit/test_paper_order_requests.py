from __future__ import annotations

import pytest

from app.paper.order_requests import _as_positive_int, build_paper_order_request_from_intent


def _candidate_intent(side: str = "long") -> dict[str, object]:
    return {
        "intent_id": "pint_test_001",
        "model_run_id": "m1",
        "signal_id": "sig_001",
        "symbol": "BTCUSDT",
        "side": side,
        "status": "candidate",
        "approved_for_paper_demo": True,
        "source_signal": {
            "entry_ref": 100.0,
            "stop_ref": 95.0 if side == "long" else 105.0,
            "take_ref": 110.0 if side == "long" else 90.0,
            "ttl_sec": 1800,
        },
        "risk_preview": {"execution_blocked": True},
        "warnings": [],
    }


def test_blocked_intent_cannot_become_ready_order_request() -> None:
    intent = _candidate_intent()
    intent["status"] = "blocked"
    request = build_paper_order_request_from_intent(intent_payload=intent)
    assert request["status"] == "blocked"
    assert any("candidate intents" in reason for reason in request["blocking_reasons"])


def test_candidate_long_intent_maps_to_buy_open_long() -> None:
    request = build_paper_order_request_from_intent(intent_payload=_candidate_intent("long"))
    assert request["status"] == "ready"
    assert request["order_side"] == "buy_open_long"
    assert request["qty"] == pytest.approx(1.0)


def test_candidate_short_intent_maps_to_sell_open_short() -> None:
    request = build_paper_order_request_from_intent(intent_payload=_candidate_intent("short"))
    assert request["status"] == "ready"
    assert request["order_side"] == "sell_open_short"
    assert request["qty"] == pytest.approx(1.0)


def test_notional_override_changes_qty() -> None:
    request = build_paper_order_request_from_intent(
        intent_payload=_candidate_intent("long"),
        execution_config={"notional": 250.0},
    )
    assert request["status"] == "ready"
    assert request["notional"] == pytest.approx(250.0)
    assert request["qty"] == pytest.approx(2.5)


def test_as_positive_int_accepts_integer_like_values() -> None:
    assert _as_positive_int(1800) == 1800
    assert _as_positive_int(1800.0) == 1800
    assert _as_positive_int("1800") == 1800


def test_as_positive_int_rejects_fractional_and_invalid_values() -> None:
    assert _as_positive_int(0) is None
    assert _as_positive_int(0.5) is None
    assert _as_positive_int("0.5") is None
    assert _as_positive_int(-1) is None
    assert _as_positive_int(True) is None
    assert _as_positive_int("") is None


def test_invalid_ttl_blocks_order_request() -> None:
    intent = _candidate_intent("long")
    source_signal = intent["source_signal"]
    assert isinstance(source_signal, dict)
    source_signal["ttl_sec"] = "0.5"
    request = build_paper_order_request_from_intent(intent_payload=intent)
    assert request["status"] == "blocked"
    assert any("ttl_sec" in reason for reason in request["blocking_reasons"])
