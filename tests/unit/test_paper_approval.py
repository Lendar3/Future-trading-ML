from __future__ import annotations

import pytest

from app.core.safety import LiveTradingDisabledError
from app.execution.real_disabled import RealBrokerInterface
from app.paper.approval import build_paper_demo_approval_report


def _readiness(
    *,
    status: str,
    score: int,
    blocking_reasons: list[str] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, object]:
    return {
        "model_run_id": "m1",
        "readiness_status": status,
        "readiness_score": score,
        "blocking_reasons": list(blocking_reasons or []),
        "warnings": list(warnings or []),
    }


def test_missing_readiness_report_is_blocked() -> None:
    report = build_paper_demo_approval_report(model_run_id="m1", readiness_report=None)
    assert report["paper_demo_status"] == "blocked"
    assert report["approved_for_paper_demo"] is False
    assert any("Readiness report is missing." in r for r in report["blocking_reasons"])


def test_not_ready_status_is_blocked() -> None:
    report = build_paper_demo_approval_report(
        model_run_id="m1",
        readiness_report=_readiness(status="not_ready", score=20),
    )
    assert report["paper_demo_status"] == "blocked"
    assert report["approved_for_paper_demo"] is False


def test_research_only_status_is_watch_only() -> None:
    report = build_paper_demo_approval_report(
        model_run_id="m1",
        readiness_report=_readiness(status="research_only", score=65),
    )
    assert report["paper_demo_status"] == "watch_only"
    assert report["approved_for_paper_demo"] is False


def test_paper_candidate_without_blockers_is_approved() -> None:
    report = build_paper_demo_approval_report(
        model_run_id="m1",
        readiness_report=_readiness(status="paper_candidate", score=88),
    )
    assert report["paper_demo_status"] == "paper_demo_candidate"
    assert report["approved_for_paper_demo"] is True
    assert report["approval_score"] == 88


def test_readiness_warnings_are_surfaced() -> None:
    report = build_paper_demo_approval_report(
        model_run_id="m1",
        readiness_report=_readiness(
            status="paper_candidate",
            score=84,
            warnings=["Top feature dominance warning."],
        ),
    )
    assert report["paper_demo_status"] == "paper_demo_candidate"
    assert any("Top feature dominance warning." in w for w in report["warnings"])


def test_live_execution_remains_disabled() -> None:
    broker = RealBrokerInterface()
    with pytest.raises(LiveTradingDisabledError):
        broker.place_order({"symbol": "BTCUSDT", "side": "Buy", "qty": 0.001})
