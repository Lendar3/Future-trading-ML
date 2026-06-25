from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

DEFAULT_NOTIONAL = 100.0


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _as_positive_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = float(text)
        except ValueError:
            return None
    else:
        return None
    if parsed <= 0.0:
        return None
    return parsed


def _as_positive_int(value: object) -> int | None:
    parsed = _as_positive_float(value)
    if parsed is None:
        return None
    if not parsed.is_integer():
        return None
    return int(parsed)


def _deterministic_id(prefix: str, *, intent_id: str, suffix: str) -> str:
    digest = hashlib.sha256(f"{intent_id}|{suffix}".encode()).hexdigest()[:24]
    return f"{prefix}_{digest}"


def build_paper_order_request_from_intent(
    *,
    intent_payload: dict[str, Any],
    execution_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blocking_reasons: list[str] = []
    warnings: list[str] = []

    intent_id = str(intent_payload.get("intent_id", "")).strip()
    model_run_id = str(intent_payload.get("model_run_id", "")).strip()
    signal_id = str(intent_payload.get("signal_id", "")).strip()
    symbol = str(intent_payload.get("symbol", "")).strip()
    side = str(intent_payload.get("side", "")).strip().lower()
    intent_status = str(intent_payload.get("status", "")).strip().lower()
    approved_for_paper_demo = bool(intent_payload.get("approved_for_paper_demo", False))

    source_signal = intent_payload.get("source_signal")
    source_signal_payload = dict(source_signal) if isinstance(source_signal, dict) else {}

    risk_preview = intent_payload.get("risk_preview")
    if not isinstance(risk_preview, dict):
        risk_preview = {}

    if intent_status != "candidate":
        blocking_reasons.append("Only candidate intents can be simulated.")
    if not approved_for_paper_demo:
        blocking_reasons.append("Intent is not approved for paper/demo.")
    if side == "no_trade":
        blocking_reasons.append("Intent side no_trade cannot be simulated.")
    if side not in {"long", "short"}:
        blocking_reasons.append(f"Intent side is invalid for simulation: {side}.")

    entry_ref = _as_positive_float(source_signal_payload.get("entry_ref"))
    stop_ref = _as_positive_float(source_signal_payload.get("stop_ref"))
    take_ref = _as_positive_float(source_signal_payload.get("take_ref"))
    ttl_sec = _as_positive_int(source_signal_payload.get("ttl_sec"))

    if entry_ref is None:
        entry_ref = _as_positive_float(risk_preview.get("entry_ref"))
    if stop_ref is None:
        stop_ref = _as_positive_float(risk_preview.get("stop_ref"))
    if take_ref is None:
        take_ref = _as_positive_float(risk_preview.get("take_ref"))
    if ttl_sec is None:
        ttl_sec = _as_positive_int(risk_preview.get("ttl_sec"))

    if entry_ref is None:
        blocking_reasons.append("Intent entry_ref is missing or invalid.")
    if stop_ref is None:
        blocking_reasons.append("Intent stop_ref is missing or invalid.")
    if take_ref is None:
        blocking_reasons.append("Intent take_ref is missing or invalid.")
    if ttl_sec is None:
        blocking_reasons.append("Intent ttl_sec is missing or invalid.")

    if (
        side == "long"
        and entry_ref is not None
        and stop_ref is not None
        and take_ref is not None
        and not (stop_ref < entry_ref < take_ref)
    ):
        blocking_reasons.append("Long intent ordering invalid: stop_ref < entry_ref < take_ref.")
    if (
        side == "short"
        and entry_ref is not None
        and stop_ref is not None
        and take_ref is not None
        and not (take_ref < entry_ref < stop_ref)
    ):
        blocking_reasons.append("Short intent ordering invalid: take_ref < entry_ref < stop_ref.")

    config = execution_config or {}
    notional = _as_positive_float(config.get("notional", DEFAULT_NOTIONAL))
    if notional is None:
        notional = DEFAULT_NOTIONAL
        warnings.append("Invalid notional in execution_config; default notional applied.")

    qty: float | None = None
    if entry_ref is not None:
        qty = notional / entry_ref
        if qty <= 0.0:
            qty = None
            blocking_reasons.append("Calculated quantity must be positive.")
    else:
        blocking_reasons.append("Cannot calculate quantity without valid entry_ref.")

    order_side = ""
    if side == "long":
        order_side = "buy_open_long"
    elif side == "short":
        order_side = "sell_open_short"

    status = "ready" if not blocking_reasons else "blocked"
    created_at = _utc_now_iso()
    order_id = _deterministic_id("pord", intent_id=intent_id, suffix="order")
    fill_id = _deterministic_id("pfill", intent_id=intent_id, suffix="fill")

    intent_warnings = intent_payload.get("warnings", [])
    if isinstance(intent_warnings, list):
        warnings.extend(str(item) for item in intent_warnings)

    return {
        "intent_id": intent_id,
        "model_run_id": model_run_id,
        "signal_id": signal_id,
        "symbol": symbol,
        "side": side,
        "order_side": order_side,
        "entry_ref": entry_ref,
        "stop_ref": stop_ref,
        "take_ref": take_ref,
        "ttl_sec": ttl_sec,
        "qty": qty,
        "notional": notional,
        "status": status,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "created_at": created_at,
        "source_intent": dict(intent_payload),
        "order_id": order_id,
        "fill_id": fill_id,
    }
