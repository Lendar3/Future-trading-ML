from __future__ import annotations

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


def build_paper_demo_approval_report(
    *,
    model_run_id: str,
    readiness_report: dict[str, Any] | None,
    required_readiness_status: str = "paper_candidate",
) -> dict[str, Any]:
    blocking_reasons: list[str] = []
    warnings: list[str] = []
    readiness_status = "not_ready"
    readiness_score = 0
    readiness_blocking_reasons: list[str] = []

    if readiness_report is None:
        blocking_reasons.append("Readiness report is missing.")
    else:
        readiness_status = str(readiness_report.get("readiness_status", "not_ready"))
        readiness_score = int(_as_float(readiness_report.get("readiness_score", 0)))
        readiness_blocking_reasons = [
            str(reason) for reason in readiness_report.get("blocking_reasons", [])
        ]
        warnings = [str(warning) for warning in readiness_report.get("warnings", [])]

    approval_checks: dict[str, dict[str, object]] = {
        "readiness_report_available": {
            "passed": readiness_report is not None,
            "detail": f"available={readiness_report is not None}",
        },
        "required_readiness_status_met": {
            "passed": readiness_status == required_readiness_status,
            "detail": (f"required={required_readiness_status}, " f"actual={readiness_status}"),
        },
        "readiness_has_no_blockers": {
            "passed": len(readiness_blocking_reasons) == 0,
            "detail": f"blocking_reasons={len(readiness_blocking_reasons)}",
        },
    }

    if readiness_status == "not_ready":
        blocking_reasons.append("Readiness status is not_ready.")
    elif readiness_status == "research_only":
        blocking_reasons.append("Readiness status is research_only.")

    if readiness_report is not None:
        blocking_reasons.extend(readiness_blocking_reasons)

    if readiness_report is None or readiness_status == "not_ready":
        paper_demo_status = "blocked"
        approved_for_paper_demo = False
        approval_score = 0
    elif readiness_status == "research_only":
        paper_demo_status = "watch_only"
        approved_for_paper_demo = False
        approval_score = min(readiness_score, 79)
    elif readiness_status == "paper_candidate" and not blocking_reasons:
        paper_demo_status = "paper_demo_candidate"
        approved_for_paper_demo = True
        approval_score = readiness_score
    else:
        paper_demo_status = "blocked"
        approved_for_paper_demo = False
        approval_score = max(0, readiness_score - 20)

    return {
        "model_run_id": model_run_id,
        "paper_demo_status": paper_demo_status,
        "approved_for_paper_demo": approved_for_paper_demo,
        "approval_score": int(approval_score),
        "required_readiness_status": required_readiness_status,
        "readiness_status": readiness_status,
        "readiness_score": int(readiness_score),
        "approval_checks": approval_checks,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "created_at": datetime.now(UTC).isoformat(),
    }
