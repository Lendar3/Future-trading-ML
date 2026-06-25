from __future__ import annotations

from typing import Any


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


def _entry_side_from_fill_side(side: str) -> str:
    normalized = side.strip().lower()
    if "long" in normalized or normalized.startswith("buy"):
        return "long"
    if "short" in normalized or normalized.startswith("sell"):
        return "short"
    raise ValueError(f"Unsupported paper fill side: {side!r}")


def apply_paper_fill_to_positions(
    *,
    current_positions: dict[str, Any] | None,
    fill_payload: dict[str, Any],
) -> dict[str, Any]:
    positions = dict(current_positions or {})

    symbol = str(fill_payload.get("symbol", "")).strip()
    side_raw = str(fill_payload.get("side", "")).strip()
    quantity = _as_positive_float(fill_payload.get("quantity"))
    entry_price = _as_positive_float(fill_payload.get("entry_price", fill_payload.get("price")))
    opened_at = str(fill_payload.get("entry_time", fill_payload.get("filled_at", ""))).strip()
    entry_fill_id = str(fill_payload.get("fill_id", "")).strip()

    if not symbol:
        raise ValueError("fill symbol is required")
    if not side_raw:
        raise ValueError("fill side is required")
    if quantity is None:
        raise ValueError("fill quantity must be positive")
    if entry_price is None:
        raise ValueError("fill entry_price must be positive")
    if not opened_at:
        raise ValueError("fill entry_time is required")
    if not entry_fill_id:
        raise ValueError("fill_id is required")

    side = _entry_side_from_fill_side(side_raw)
    existing = positions.get(symbol)
    if not isinstance(existing, dict):
        existing = None

    if existing is not None and str(existing.get("status", "")).lower() == "open":
        existing_side = str(existing.get("side", "")).strip().lower()
        if existing_side and existing_side != side:
            return positions
        existing_qty = _as_positive_float(existing.get("quantity"))
        existing_entry = _as_positive_float(existing.get("entry_price"))
        if existing_qty is not None and existing_entry is not None:
            merged_qty = existing_qty + quantity
            merged_entry = ((existing_qty * existing_entry) + (quantity * entry_price)) / merged_qty
            positions[symbol] = {
                "symbol": symbol,
                "side": side,
                "quantity": merged_qty,
                "entry_price": merged_entry,
                "notional": merged_qty * merged_entry,
                "opened_at": str(existing.get("opened_at", opened_at)),
                "entry_fill_id": str(existing.get("entry_fill_id", entry_fill_id)),
                "status": "open",
            }
            return positions

    positions[symbol] = {
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "entry_price": entry_price,
        "notional": quantity * entry_price,
        "opened_at": opened_at,
        "entry_fill_id": entry_fill_id,
        "status": "open",
    }
    return positions


def apply_paper_close_to_positions(
    *,
    current_positions: dict[str, Any] | None,
    close_payload: dict[str, Any],
) -> dict[str, Any]:
    positions = dict(current_positions or {})
    symbol = str(close_payload.get("symbol", "")).strip()
    side_raw = str(close_payload.get("side", "")).strip()
    entry_fill_id = str(close_payload.get("entry_fill_id", "")).strip()

    if not symbol or not side_raw:
        return positions

    existing = positions.get(symbol)
    if not isinstance(existing, dict):
        return positions
    if str(existing.get("status", "")).strip().lower() != "open":
        return positions

    try:
        close_side = _entry_side_from_fill_side(side_raw)
    except ValueError:
        return positions

    existing_side = str(existing.get("side", "")).strip().lower()
    if existing_side and existing_side != close_side:
        return positions

    existing_entry_fill_id = str(existing.get("entry_fill_id", "")).strip()
    if existing_entry_fill_id and entry_fill_id and existing_entry_fill_id != entry_fill_id:
        return positions

    positions.pop(symbol, None)
    return positions
