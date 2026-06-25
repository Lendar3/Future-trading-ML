from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def _as_float_or_none(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _parse_iso_datetime(value: object) -> str | None:
    parsed = _parse_datetime(value)
    return parsed.isoformat() if parsed is not None else None


def _parse_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        dt = parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    return None


def _side_multiplier(side: str) -> float:
    side_lower = side.strip().lower()
    if "short" in side_lower or side_lower.startswith("sell"):
        return -1.0
    return 1.0


def build_paper_fill_accounting(
    *,
    fill_payload: dict[str, Any],
    order_payload: dict[str, Any] | None = None,
    source_payload: dict[str, Any] | None = None,
    exit_price: float | None = None,
    mark_price: float | None = None,
) -> dict[str, Any]:
    fill = dict(fill_payload)
    order = dict(order_payload or {})
    source = dict(source_payload or {})

    quantity = _as_float_or_none(fill.get("quantity"))
    entry_price = _as_float_or_none(fill.get("price"))
    fee = _as_float_or_none(fill.get("fee"))

    if fee is None:
        fee = _as_float_or_none(source.get("fee"))
    if fee is None:
        fee = 0.0

    notional = None
    if quantity is not None and entry_price is not None:
        notional = abs(quantity * entry_price)

    side = str(fill.get("side", order.get("side", source.get("side", ""))))
    direction = _side_multiplier(side)

    parsed_entry_time = _parse_iso_datetime(fill.get("filled_at"))
    if parsed_entry_time is None:
        parsed_entry_time = _parse_iso_datetime(source.get("entry_time"))
    parsed_exit_time = _parse_iso_datetime(source.get("exit_time"))

    resolved_mark = (
        mark_price if mark_price is not None else _as_float_or_none(source.get("mark_price"))
    )
    resolved_exit = (
        exit_price if exit_price is not None else _as_float_or_none(source.get("exit_price"))
    )

    source_status = str(source.get("pnl_status", "")).strip().lower()
    pnl_status = "open"
    realized_pnl: float | None = None
    unrealized_pnl: float | None = None
    close_reason = source.get("close_reason")
    if close_reason is not None:
        close_reason = str(close_reason)
    resolved_fee_paid = _as_float_or_none(source.get("fees_paid"))
    if resolved_fee_paid is None:
        resolved_total_fees = _as_float_or_none(source.get("total_fees_paid"))
        if resolved_total_fees is not None:
            resolved_fee_paid = resolved_total_fees
    if resolved_fee_paid is not None:
        fee = resolved_fee_paid

    if quantity is None or entry_price is None:
        pnl_status = "unavailable"
    elif source_status in {"open", "closed", "unavailable"}:
        pnl_status = source_status
        realized_pnl = _as_float_or_none(source.get("realized_pnl"))
        unrealized_pnl = _as_float_or_none(source.get("unrealized_pnl"))
        if pnl_status == "closed" and realized_pnl is None and resolved_exit is not None:
            realized_pnl = (resolved_exit - entry_price) * quantity * direction
    elif resolved_exit is not None:
        pnl_status = "closed"
        realized_pnl = (resolved_exit - entry_price) * quantity * direction
    elif resolved_mark is not None:
        unrealized_pnl = (resolved_mark - entry_price) * quantity * direction

    return {
        "notional": notional,
        "fees_paid": fee,
        "entry_price": entry_price,
        "entry_time": parsed_entry_time,
        "exit_price": resolved_exit,
        "exit_time": parsed_exit_time,
        "close_reason": close_reason,
        "pnl_status": pnl_status,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
    }


def build_paper_close_accounting(
    *,
    entry_fill_payload: dict[str, Any],
    exit_price: float,
    exit_time: datetime | str,
    exit_fee: float = 0.0,
    close_reason: str = "manual_exit",
) -> dict[str, Any]:
    fill = dict(entry_fill_payload)

    entry_fill_id = str(fill.get("fill_id", "")).strip()
    if not entry_fill_id:
        raise ValueError("entry fill_id is required")

    side = str(fill.get("side", "")).strip()
    if not side:
        raise ValueError("entry fill side is required")

    quantity = _as_float_or_none(fill.get("quantity"))
    if quantity is None or quantity <= 0.0:
        raise ValueError("entry fill quantity must be positive")

    entry_price = _as_float_or_none(fill.get("entry_price", fill.get("price")))
    if entry_price is None or entry_price <= 0.0:
        raise ValueError("entry fill entry_price must be positive")

    parsed_entry_time = _parse_datetime(fill.get("entry_time", fill.get("filled_at")))
    if parsed_entry_time is None:
        raise ValueError("entry fill entry_time is required")

    parsed_exit_time = _parse_datetime(exit_time)
    if parsed_exit_time is None:
        raise ValueError("exit_time must be a valid ISO datetime or datetime object")
    if parsed_exit_time < parsed_entry_time:
        raise ValueError("exit_time must be greater than or equal to entry_time")

    resolved_exit_price = _as_float_or_none(exit_price)
    if resolved_exit_price is None or resolved_exit_price <= 0.0:
        raise ValueError("exit_price must be positive")

    resolved_exit_fee = _as_float_or_none(exit_fee)
    if resolved_exit_fee is None:
        raise ValueError("exit_fee must be a valid non-negative number")
    if resolved_exit_fee < 0.0:
        raise ValueError("exit_fee must be non-negative")

    entry_fee = _as_float_or_none(fill.get("fees_paid", fill.get("fee")))
    if entry_fee is None:
        entry_fee = 0.0
    if entry_fee < 0.0:
        raise ValueError("entry fee must be non-negative")

    direction = _side_multiplier(side)
    gross_pnl = (resolved_exit_price - entry_price) * quantity * direction
    total_fees_paid = entry_fee + resolved_exit_fee
    realized_pnl = gross_pnl - total_fees_paid

    return {
        "entry_fill_id": entry_fill_id,
        "side": side,
        "quantity": quantity,
        "entry_price": entry_price,
        "exit_price": resolved_exit_price,
        "entry_time": parsed_entry_time.isoformat(),
        "exit_time": parsed_exit_time.isoformat(),
        "gross_pnl": gross_pnl,
        "total_fees_paid": total_fees_paid,
        "realized_pnl": realized_pnl,
        "pnl_status": "closed",
        "close_reason": close_reason.strip() or "manual_exit",
        "notional": abs(quantity * entry_price),
    }
