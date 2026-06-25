from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

ALLOWED_SIDES = {"long", "short", "no_trade"}


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _coerce_event_time(value: object) -> tuple[str, bool]:
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return dt.astimezone(UTC).isoformat(), True
    if isinstance(value, str) and value.strip():
        text = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return _utc_now_iso(), False
        dt = parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
        return dt.astimezone(UTC).isoformat(), True
    return _utc_now_iso(), False


def _as_positive_float(value: object) -> tuple[float | None, bool]:
    if isinstance(value, bool):
        return None, False
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None, False
        try:
            parsed = float(text)
        except ValueError:
            return None, False
    else:
        return None, False
    if parsed <= 0.0:
        return parsed, False
    return parsed, True


def _risk_reward_preview(
    *,
    side: str,
    entry_ref: float | None,
    stop_ref: float | None,
    take_ref: float | None,
) -> tuple[float | None, float | None, float | None]:
    if entry_ref is None or stop_ref is None or take_ref is None:
        return None, None, None
    if side == "long":
        risk = (entry_ref - stop_ref) / entry_ref
        reward = (take_ref - entry_ref) / entry_ref
    elif side == "short":
        risk = (stop_ref - entry_ref) / entry_ref
        reward = (entry_ref - take_ref) / entry_ref
    else:
        return None, None, None
    reward_to_risk = None if risk <= 0.0 else reward / risk
    return risk, reward, reward_to_risk


def _intent_id(
    *,
    model_run_id: str,
    signal_id: str,
    symbol: str,
    side: str,
    event_time: str,
) -> str:
    fingerprint = hashlib.sha256(
        f"{model_run_id}|{signal_id}|{symbol}|{side}|{event_time}".encode()
    ).hexdigest()[:24]
    return f"pint_{fingerprint}"


def build_paper_intent_candidate(
    *,
    model_run_id: str,
    signal_payload: dict[str, Any],
    approval_report: dict[str, Any] | None,
    risk_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blocking_reasons: list[str] = []
    warnings: list[str] = []

    signal_id = str(signal_payload.get("signal_id", "")).strip()
    symbol = str(signal_payload.get("symbol", "")).strip()
    side = str(signal_payload.get("side", "")).strip().lower()
    event_time, has_valid_event_time = _coerce_event_time(signal_payload.get("event_time"))
    entry_ref, entry_ref_ok = _as_positive_float(signal_payload.get("entry_ref"))
    stop_ref, stop_ref_ok = _as_positive_float(signal_payload.get("stop_ref"))
    take_ref, take_ref_ok = _as_positive_float(signal_payload.get("take_ref"))
    ttl_sec, ttl_sec_ok = _as_positive_float(signal_payload.get("ttl_sec"))
    rationale_json = signal_payload.get("rationale_json")
    if not isinstance(rationale_json, dict):
        rationale_json = {}

    required_missing: list[str] = []
    if not signal_id:
        required_missing.append("signal_id")
    if not symbol:
        required_missing.append("symbol")
    if not side:
        required_missing.append("side")
    if required_missing:
        blocking_reasons.append(
            "Missing required signal fields: " + ", ".join(sorted(required_missing)) + "."
        )
    if not has_valid_event_time:
        blocking_reasons.append("Signal event_time is missing or invalid.")
    if side and side not in ALLOWED_SIDES:
        blocking_reasons.append(f"Signal side is invalid: {side}.")

    approved_for_paper_demo = False
    if approval_report is None:
        blocking_reasons.append("Approval report is missing.")
    else:
        approved_for_paper_demo = bool(approval_report.get("approved_for_paper_demo", False))
        if not approved_for_paper_demo:
            blocking_reasons.append("Model is not approved for paper/demo.")
        approval_warnings = approval_report.get("warnings", [])
        if isinstance(approval_warnings, list):
            warnings.extend(str(item) for item in approval_warnings)

    if side == "no_trade":
        blocking_reasons.append("Signal side is no_trade.")

    if not entry_ref_ok:
        blocking_reasons.append("Signal entry_ref must be numeric and > 0.")
    if not stop_ref_ok:
        blocking_reasons.append("Signal stop_ref must be numeric and > 0.")
    if not take_ref_ok:
        blocking_reasons.append("Signal take_ref must be numeric and > 0.")
    if not ttl_sec_ok:
        blocking_reasons.append("Signal ttl_sec must be numeric and > 0.")

    if (
        entry_ref_ok
        and stop_ref_ok
        and take_ref_ok
        and side == "long"
        and entry_ref is not None
        and stop_ref is not None
        and take_ref is not None
        and not (stop_ref < entry_ref < take_ref)
    ):
        blocking_reasons.append(
            "Long signal ordering is invalid: expected stop_ref < entry_ref < take_ref."
        )
    if (
        entry_ref_ok
        and stop_ref_ok
        and take_ref_ok
        and side == "short"
        and entry_ref is not None
        and stop_ref is not None
        and take_ref is not None
        and not (take_ref < entry_ref < stop_ref)
    ):
        blocking_reasons.append(
            "Short signal ordering is invalid: expected take_ref < entry_ref < stop_ref."
        )

    status = "candidate"
    if blocking_reasons:
        status = "blocked"

    created_at = _utc_now_iso()
    intent_id = _intent_id(
        model_run_id=model_run_id,
        signal_id=signal_id or "missing-signal-id",
        symbol=symbol or "missing-symbol",
        side=side or "missing-side",
        event_time=event_time,
    )
    estimated_risk_pct, estimated_reward_pct, reward_to_risk = _risk_reward_preview(
        side=side,
        entry_ref=entry_ref,
        stop_ref=stop_ref,
        take_ref=take_ref,
    )

    risk_preview: dict[str, Any] = {
        "preview_only": True,
        "execution_blocked": True,
        "side": side,
        "entry_ref": entry_ref,
        "stop_ref": stop_ref,
        "take_ref": take_ref,
        "ttl_sec": ttl_sec,
        "estimated_risk_pct": estimated_risk_pct,
        "estimated_reward_pct": estimated_reward_pct,
        "reward_to_risk": reward_to_risk,
        "risk_config_applied": bool(risk_config),
    }
    if risk_config is not None:
        risk_preview["risk_config"] = dict(risk_config)

    return {
        "intent_id": intent_id,
        "model_run_id": model_run_id,
        "signal_id": signal_id,
        "symbol": symbol,
        "side": side,
        "event_time": event_time,
        "status": status,
        "approved_for_paper_demo": approved_for_paper_demo,
        "risk_preview": risk_preview,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "created_at": created_at,
        "source_signal": dict(signal_payload),
        "rationale_json": rationale_json,
    }
