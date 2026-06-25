from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


@dataclass(frozen=True)
class _WeightedCheck:
    key: str
    passed: bool
    points: int
    detail: str


def build_research_readiness_report(
    *,
    model_run_id: str,
    model_evaluation_payload: dict[str, Any] | None,
    feature_importance_payload: dict[str, Any] | None,
    strategy_comparison_payload: dict[str, Any] | None = None,
    extra_warnings: list[str] | None = None,
) -> dict[str, Any]:
    warnings: list[str] = list(extra_warnings or [])
    blocking_reasons: list[str] = []
    hard_blockers: list[str] = []

    evaluation = model_evaluation_payload or {}
    importance = feature_importance_payload or {}
    strategy = strategy_comparison_payload or {}

    eval_metrics = evaluation.get("metrics", {}) if isinstance(evaluation, dict) else {}
    eval_baseline = (
        evaluation.get("baseline_comparison", {}) if isinstance(evaluation, dict) else {}
    )
    eval_class = evaluation.get("class_distribution", {}) if isinstance(evaluation, dict) else {}

    test_accuracy = _as_float(eval_metrics.get("test_accuracy"))
    baseline_majority = _as_float(eval_baseline.get("baseline_majority_accuracy"))
    baseline_random = _as_float(eval_baseline.get("baseline_random_accuracy"))
    rows_test = int(_as_float(evaluation.get("rows_test", 0)))
    max_class_ratio = _as_float(eval_class.get("target_max_class_ratio"))
    high_imbalance_flag = int(_as_float(eval_class.get("target_high_imbalance_flag")))

    top_features = importance.get("top_features", []) if isinstance(importance, dict) else []
    top_feature_share = _as_float(top_features[0].get("importance_share")) if top_features else 0.0
    feature_count = int(_as_float(importance.get("feature_count", 0)))
    importance_warning = str(importance.get("warning", "")).strip()

    strategy_quality_gates = strategy.get("quality_gates", {}) if isinstance(strategy, dict) else {}
    strategy_quality_gate_passed = bool(strategy.get("quality_gate_passed", False))
    strategy_available = bool(strategy_comparison_payload)

    checks: list[_WeightedCheck] = [
        _WeightedCheck(
            "model_beats_majority_baseline",
            test_accuracy > baseline_majority,
            10,
            f"test_accuracy={test_accuracy:.4f}, baseline_majority={baseline_majority:.4f}",
        ),
        _WeightedCheck(
            "model_beats_random_baseline",
            test_accuracy > baseline_random,
            10,
            f"test_accuracy={test_accuracy:.4f}, baseline_random={baseline_random:.4f}",
        ),
        _WeightedCheck(
            "target_not_extremely_imbalanced",
            high_imbalance_flag == 0,
            10,
            (
                "target_max_class_ratio="
                f"{max_class_ratio:.4f}, high_imbalance_flag={high_imbalance_flag}"
            ),
        ),
        _WeightedCheck(
            "test_rows_positive",
            rows_test > 0,
            10,
            f"rows_test={rows_test}",
        ),
        _WeightedCheck(
            "feature_count_positive",
            feature_count > 0,
            10,
            f"feature_count={feature_count}",
        ),
        _WeightedCheck(
            "top_feature_not_dominant",
            top_feature_share <= 0.50,
            10,
            f"top_feature_importance_share={top_feature_share:.4f}",
        ),
        _WeightedCheck(
            "feature_importance_warning_absent",
            importance_warning == "",
            10,
            f"warning={importance_warning or 'none'}",
        ),
        _WeightedCheck(
            "strategy_quality_gate_passed",
            strategy_available and strategy_quality_gate_passed,
            8,
            f"available={strategy_available}, quality_gate_passed={strategy_quality_gate_passed}",
        ),
        _WeightedCheck(
            "strategy_ml_has_trades",
            strategy_available and bool(strategy_quality_gates.get("ml_has_trades", False)),
            6,
            f"ml_has_trades={strategy_quality_gates.get('ml_has_trades', False)}",
        ),
        _WeightedCheck(
            "strategy_ml_beats_best_rule_total_return",
            strategy_available
            and bool(strategy_quality_gates.get("ml_beats_best_rule_total_return", False)),
            6,
            "ml_beats_best_rule_total_return="
            f"{strategy_quality_gates.get('ml_beats_best_rule_total_return', False)}",
        ),
        _WeightedCheck(
            "strategy_ml_beats_best_rule_profit_factor",
            strategy_available
            and bool(strategy_quality_gates.get("ml_beats_best_rule_profit_factor", False)),
            6,
            "ml_beats_best_rule_profit_factor="
            f"{strategy_quality_gates.get('ml_beats_best_rule_profit_factor', False)}",
        ),
        _WeightedCheck(
            "strategy_enough_folds",
            strategy_available and bool(strategy_quality_gates.get("enough_folds", False)),
            6,
            f"enough_folds={strategy_quality_gates.get('enough_folds', False)}",
        ),
        _WeightedCheck(
            "strategy_positive_fold_ratio_ok",
            strategy_available
            and bool(strategy_quality_gates.get("positive_fold_ratio_ok", False)),
            8,
            f"positive_fold_ratio_ok={strategy_quality_gates.get('positive_fold_ratio_ok', False)}",
        ),
    ]

    if model_evaluation_payload is None:
        hard_blockers.append("Model evaluation artifact is missing.")
    if feature_importance_payload is None:
        hard_blockers.append("Feature importance summary is missing.")
    if rows_test <= 0:
        hard_blockers.append("Model evaluation has no test rows.")
    if high_imbalance_flag == 1:
        blocking_reasons.append("Target distribution is highly imbalanced.")
    if not strategy_available:
        blocking_reasons.append("Strategy comparison data is missing.")
        warnings.append("Strategy comparison payload is unavailable; paper_candidate is blocked.")
    if importance_warning:
        warnings.append(f"Feature importance warning: {importance_warning}")
    if top_feature_share > 0.50:
        warnings.append(
            "Top feature dominance is high "
            f"({top_feature_share:.2%}); investigate feature reliance."
        )
        blocking_reasons.append("Top feature dominance exceeds the readiness threshold.")
    if not strategy_quality_gate_passed and strategy_available:
        blocking_reasons.append("Strategy comparison quality gates did not pass.")

    max_points = sum(check.points for check in checks)
    points = sum(check.points for check in checks if check.passed)
    readiness_score = int(round((points / max_points) * 100)) if max_points > 0 else 0

    blocking_reasons = hard_blockers + blocking_reasons
    readiness_status = "not_ready"
    if hard_blockers:
        readiness_status = "not_ready"
    elif readiness_score >= 80 and not blocking_reasons:
        readiness_status = "paper_candidate"
    elif readiness_score >= 50:
        readiness_status = "research_only"
    else:
        readiness_status = "not_ready"

    readiness_checks: dict[str, dict[str, object]] = {}
    for check in checks:
        readiness_checks[check.key] = {
            "passed": check.passed,
            "points": check.points,
            "detail": check.detail,
        }

    model_evaluation_summary: dict[str, Any] = {
        "available": model_evaluation_payload is not None,
        "test_accuracy": test_accuracy,
        "baseline_majority_accuracy": baseline_majority,
        "baseline_random_accuracy": baseline_random,
        "rows_test": rows_test,
        "target_max_class_ratio": max_class_ratio,
        "target_high_imbalance_flag": high_imbalance_flag,
    }
    feature_importance_summary: dict[str, Any] = {
        "available": feature_importance_payload is not None,
        "feature_count": feature_count,
        "top_feature": top_features[0].get("feature") if top_features else "",
        "top_feature_importance_share": top_feature_share,
        "warning": importance_warning,
    }
    strategy_comparison_summary: dict[str, Any] = {
        "available": strategy_available,
        "quality_gate_passed": strategy_quality_gate_passed if strategy_available else False,
        "summary": strategy.get("summary", {}) if strategy_available else {},
    }
    quality_gates_summary: dict[str, Any] = {
        "available": strategy_available,
        "quality_gates": strategy_quality_gates if strategy_available else {},
        "quality_gate_warnings": (
            strategy.get("quality_gate_warnings", []) if strategy_available else []
        ),
    }

    return {
        "model_run_id": model_run_id,
        "readiness_status": readiness_status,
        "readiness_score": readiness_score,
        "readiness_checks": readiness_checks,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "model_evaluation_summary": model_evaluation_summary,
        "feature_importance_summary": feature_importance_summary,
        "strategy_comparison_summary": strategy_comparison_summary,
        "quality_gates_summary": quality_gates_summary,
        "created_at": datetime.now(UTC).isoformat(),
    }
