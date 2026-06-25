from __future__ import annotations

import pytest

from app.core.safety import LiveTradingDisabledError
from app.execution.real_disabled import RealBrokerInterface
from app.paper.intents import build_paper_intent_candidate


def _signal_payload(side: str = "long") -> dict[str, object]:
    return {
        "signal_id": "sig_001",
        "symbol": "BTCUSDT",
        "side": side,
        "event_time": "2026-01-01T00:00:00+00:00",
        "entry_ref": 100.0,
        "stop_ref": 95.0,
        "take_ref": 110.0,
        "ttl_sec": 1800,
        "rationale_json": {"source": "unit"},
    }


def _approval(approved: bool) -> dict[str, object]:
    return {
        "model_run_id": "m1",
        "approved_for_paper_demo": approved,
        "paper_demo_status": "paper_demo_candidate" if approved else "blocked",
        "warnings": [],
    }


def test_intent_blocked_when_approval_missing() -> None:
    intent = build_paper_intent_candidate(
        model_run_id="m1",
        signal_payload=_signal_payload(),
        approval_report=None,
    )
    assert intent["status"] == "blocked"
    assert any("Approval report is missing." in r for r in intent["blocking_reasons"])


def test_intent_blocked_when_approval_not_approved() -> None:
    intent = build_paper_intent_candidate(
        model_run_id="m1",
        signal_payload=_signal_payload(),
        approval_report=_approval(approved=False),
    )
    assert intent["status"] == "blocked"
    assert intent["approved_for_paper_demo"] is False


def test_intent_blocked_for_no_trade_signal() -> None:
    intent = build_paper_intent_candidate(
        model_run_id="m1",
        signal_payload=_signal_payload(side="no_trade"),
        approval_report=_approval(approved=True),
    )
    assert intent["status"] == "blocked"
    assert any("no_trade" in reason for reason in intent["blocking_reasons"])


def test_intent_blocked_for_invalid_side() -> None:
    intent = build_paper_intent_candidate(
        model_run_id="m1",
        signal_payload=_signal_payload(side="flat"),
        approval_report=_approval(approved=True),
    )
    assert intent["status"] == "blocked"
    assert any("invalid" in reason for reason in intent["blocking_reasons"])


def test_intent_blocked_for_missing_refs() -> None:
    payload = _signal_payload()
    payload.pop("entry_ref")
    payload.pop("stop_ref")
    payload.pop("take_ref")
    payload.pop("ttl_sec")
    intent = build_paper_intent_candidate(
        model_run_id="m1",
        signal_payload=payload,
        approval_report=_approval(approved=True),
    )
    assert intent["status"] == "blocked"
    assert any("entry_ref" in reason for reason in intent["blocking_reasons"])
    assert any("stop_ref" in reason for reason in intent["blocking_reasons"])
    assert any("take_ref" in reason for reason in intent["blocking_reasons"])
    assert any("ttl_sec" in reason for reason in intent["blocking_reasons"])


def test_intent_blocked_for_non_numeric_refs() -> None:
    payload = _signal_payload()
    payload["entry_ref"] = "bad"
    payload["stop_ref"] = "bad"
    payload["take_ref"] = "bad"
    payload["ttl_sec"] = "bad"
    intent = build_paper_intent_candidate(
        model_run_id="m1",
        signal_payload=payload,
        approval_report=_approval(approved=True),
    )
    assert intent["status"] == "blocked"


def test_long_ordering_violation_blocks_intent() -> None:
    payload = _signal_payload(side="long")
    payload["stop_ref"] = 105.0
    payload["take_ref"] = 98.0
    intent = build_paper_intent_candidate(
        model_run_id="m1",
        signal_payload=payload,
        approval_report=_approval(approved=True),
    )
    assert intent["status"] == "blocked"
    assert any("Long signal ordering is invalid" in reason for reason in intent["blocking_reasons"])


def test_short_ordering_violation_blocks_intent() -> None:
    payload = _signal_payload(side="short")
    payload["entry_ref"] = 100.0
    payload["stop_ref"] = 95.0
    payload["take_ref"] = 105.0
    intent = build_paper_intent_candidate(
        model_run_id="m1",
        signal_payload=payload,
        approval_report=_approval(approved=True),
    )
    assert intent["status"] == "blocked"
    assert any(
        "Short signal ordering is invalid" in reason for reason in intent["blocking_reasons"]
    )


def test_valid_approved_long_signal_creates_candidate_intent() -> None:
    intent = build_paper_intent_candidate(
        model_run_id="m1",
        signal_payload=_signal_payload(),
        approval_report=_approval(approved=True),
    )
    assert intent["status"] == "candidate"
    assert intent["approved_for_paper_demo"] is True
    assert intent["signal_id"] == "sig_001"
    assert intent["symbol"] == "BTCUSDT"
    preview = intent["risk_preview"]
    assert preview["preview_only"] is True
    assert preview["execution_blocked"] is True
    assert preview["side"] == "long"
    assert preview["estimated_risk_pct"] == pytest.approx(0.05)
    assert preview["estimated_reward_pct"] == pytest.approx(0.10)


def test_valid_approved_short_signal_creates_candidate_intent() -> None:
    payload = _signal_payload(side="short")
    payload["entry_ref"] = 100.0
    payload["stop_ref"] = 105.0
    payload["take_ref"] = 90.0
    intent = build_paper_intent_candidate(
        model_run_id="m1",
        signal_payload=payload,
        approval_report=_approval(approved=True),
    )
    assert intent["status"] == "candidate"
    preview = intent["risk_preview"]
    assert preview["side"] == "short"
    assert preview["estimated_risk_pct"] == pytest.approx(0.05)
    assert preview["estimated_reward_pct"] == pytest.approx(0.10)


def test_reward_to_risk_is_calculated_correctly() -> None:
    intent = build_paper_intent_candidate(
        model_run_id="m1",
        signal_payload=_signal_payload(),
        approval_report=_approval(approved=True),
    )
    assert intent["risk_preview"]["reward_to_risk"] == pytest.approx(2.0)


def test_live_execution_remains_disabled() -> None:
    broker = RealBrokerInterface()
    with pytest.raises(LiveTradingDisabledError):
        broker.place_order({"symbol": "BTCUSDT", "side": "Buy", "qty": 0.001})
