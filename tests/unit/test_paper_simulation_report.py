from __future__ import annotations

from app.reports.paper_simulation import build_paper_simulation_audit_report


def test_audit_report_handles_empty_payloads() -> None:
    report = build_paper_simulation_audit_report(
        paper_intents=[],
        paper_orders=[],
        paper_fills=[],
        paper_intent_executions=[],
        portfolio_snapshot=None,
        limit=50,
    )

    assert report["summary"]["total_intents"] == 0
    assert report["summary"]["total_orders"] == 0
    assert report["summary"]["total_fills"] == 0
    assert report["summary"]["simulated_intents"] == 0
    assert report["summary"]["unsimulated_candidate_intents"] == 0
    assert report["summary"]["duplicate_replay_count"] is None


def test_audit_report_counts_blocked_and_candidate_intents() -> None:
    intents = [
        {"intent_id": "i1", "status": "candidate", "side": "long", "symbol": "BTCUSDT"},
        {"intent_id": "i2", "status": "blocked", "side": "short", "symbol": "ETHUSDT"},
        {"intent_id": "i3", "status": "blocked", "side": "no_trade", "symbol": "BTCUSDT"},
    ]
    report = build_paper_simulation_audit_report(
        paper_intents=intents,
        paper_orders=[],
        paper_fills=[],
        paper_intent_executions=[],
        portfolio_snapshot={},
    )

    assert report["summary"]["candidate_intents"] == 1
    assert report["summary"]["blocked_intents"] == 2
    assert report["summary"]["long_intents"] == 1
    assert report["summary"]["short_intents"] == 1
    assert report["summary"]["no_trade_or_invalid_intents"] == 1


def test_audit_report_counts_simulated_intents_from_executions() -> None:
    intents = [
        {"intent_id": "i1", "status": "candidate", "side": "long", "symbol": "BTCUSDT"},
        {"intent_id": "i2", "status": "candidate", "side": "short", "symbol": "ETHUSDT"},
    ]
    executions = [
        {
            "intent_id": "i1",
            "order_id": "o1",
            "fill_id": "f1",
            "created_at": "2026-01-01T00:00:00+00:00",
        }
    ]

    report = build_paper_simulation_audit_report(
        paper_intents=intents,
        paper_orders=[],
        paper_fills=[],
        paper_intent_executions=executions,
        portfolio_snapshot={},
    )

    assert report["summary"]["simulated_intents"] == 1
    assert report["simulation_status_counts"]["execution_records"] == 1
    assert report["latest_simulations"][0]["intent_id"] == "i1"


def test_audit_report_calculates_unsimulated_candidate_intents() -> None:
    intents = [
        {"intent_id": "i1", "status": "candidate", "side": "long", "symbol": "BTCUSDT"},
        {"intent_id": "i2", "status": "candidate", "side": "short", "symbol": "ETHUSDT"},
        {"intent_id": "i3", "status": "blocked", "side": "no_trade", "symbol": "SOLUSDT"},
    ]
    executions = [
        {
            "intent_id": "i1",
            "order_id": "o1",
            "fill_id": "f1",
            "created_at": "2026-01-01T00:00:00+00:00",
        }
    ]

    report = build_paper_simulation_audit_report(
        paper_intents=intents,
        paper_orders=[],
        paper_fills=[],
        paper_intent_executions=executions,
        portfolio_snapshot={},
    )

    assert report["summary"]["candidate_intents"] == 2
    assert report["summary"]["simulated_intents"] == 1
    assert report["summary"]["unsimulated_candidate_intents"] == 1


def test_audit_report_counts_blocking_reasons() -> None:
    intents = [
        {
            "intent_id": "i1",
            "status": "blocked",
            "side": "flat",
            "symbol": "BTCUSDT",
            "blocking_reasons": ["invalid side", "approval missing"],
        },
        {
            "intent_id": "i2",
            "status": "blocked",
            "side": "flat",
            "symbol": "ETHUSDT",
            "blocking_reasons": ["invalid side"],
        },
    ]

    report = build_paper_simulation_audit_report(
        paper_intents=intents,
        paper_orders=[],
        paper_fills=[],
        paper_intent_executions=[],
        portfolio_snapshot={},
    )

    assert report["blocking_reason_counts"]["invalid side"] == 2
    assert report["blocking_reason_counts"]["approval missing"] == 1
