from __future__ import annotations

import pytest

from app.core.safety import LiveTradingDisabledError
from app.execution.real_disabled import RealBrokerInterface
from app.reports.readiness import build_research_readiness_report


def _evaluation_payload() -> dict[str, object]:
    return {
        "model_run_id": "m1",
        "rows_test": 25,
        "metrics": {"test_accuracy": 0.62},
        "baseline_comparison": {
            "baseline_majority_accuracy": 0.50,
            "baseline_random_accuracy": 0.41,
        },
        "class_distribution": {
            "target_max_class_ratio": 0.65,
            "target_high_imbalance_flag": 0.0,
        },
    }


def _importance_payload(top_share: float = 0.25, warning: str = "") -> dict[str, object]:
    payload: dict[str, object] = {
        "model_run_id": "m1",
        "feature_count": 10,
        "top_features": [
            {
                "feature": "feature_a",
                "importance": 12.0,
                "rank": 1,
                "importance_share": top_share,
            }
        ],
    }
    if warning:
        payload["warning"] = warning
    return payload


def _strategy_payload(passed: bool = True) -> dict[str, object]:
    return {
        "quality_gate_passed": passed,
        "quality_gates": {
            "ml_has_trades": passed,
            "ml_beats_best_rule_total_return": passed,
            "ml_beats_best_rule_profit_factor": passed,
            "enough_folds": passed,
            "positive_fold_ratio_ok": passed,
        },
        "quality_gate_warnings": [],
        "summary": {"best_strategy_by_total_return": "ml_advisor"},
    }


def test_not_ready_when_required_data_missing() -> None:
    report = build_research_readiness_report(
        model_run_id="m1",
        model_evaluation_payload=None,
        feature_importance_payload=None,
        strategy_comparison_payload=None,
    )
    assert report["readiness_status"] == "not_ready"
    assert "Model evaluation artifact is missing." in report["blocking_reasons"]


def test_research_only_when_strategy_missing() -> None:
    report = build_research_readiness_report(
        model_run_id="m1",
        model_evaluation_payload=_evaluation_payload(),
        feature_importance_payload=_importance_payload(),
        strategy_comparison_payload=None,
    )
    assert report["readiness_status"] == "research_only"
    assert report["readiness_score"] >= 50
    assert any("Strategy comparison data is missing." in r for r in report["blocking_reasons"])


def test_paper_candidate_only_when_all_core_checks_pass() -> None:
    report = build_research_readiness_report(
        model_run_id="m1",
        model_evaluation_payload=_evaluation_payload(),
        feature_importance_payload=_importance_payload(),
        strategy_comparison_payload=_strategy_payload(passed=True),
    )
    assert report["readiness_status"] == "paper_candidate"
    assert report["readiness_score"] >= 80


def test_dominant_top_feature_creates_warning_and_prevents_paper_candidate() -> None:
    report = build_research_readiness_report(
        model_run_id="m1",
        model_evaluation_payload=_evaluation_payload(),
        feature_importance_payload=_importance_payload(top_share=0.75),
        strategy_comparison_payload=_strategy_payload(passed=True),
    )
    assert report["readiness_status"] != "paper_candidate"
    assert any("Top feature dominance is high" in w for w in report["warnings"])


def test_missing_feature_importance_prevents_paper_candidate() -> None:
    report = build_research_readiness_report(
        model_run_id="m1",
        model_evaluation_payload=_evaluation_payload(),
        feature_importance_payload=None,
        strategy_comparison_payload=_strategy_payload(passed=True),
        extra_warnings=["Feature importance artifact is unavailable for this model run."],
    )
    assert report["readiness_status"] == "not_ready"
    assert any("Feature importance summary is missing." in r for r in report["blocking_reasons"])


def test_live_execution_remains_disabled() -> None:
    broker = RealBrokerInterface()
    with pytest.raises(LiveTradingDisabledError):
        broker.place_order({"symbol": "BTCUSDT", "side": "Buy", "qty": 0.001})
