from __future__ import annotations

from datetime import UTC, datetime, timedelta
from math import isnan
from typing import Any


def _as_datetime_utc(value: object, *, field_name: str) -> datetime:
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        if not text:
            raise ValueError(f"{field_name} must be a valid ISO datetime.")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a valid ISO datetime.") from exc
        dt = parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    raise ValueError(f"{field_name} must be a valid ISO datetime.")


def _as_positive_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be positive.")
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{field_name} must be positive.")
        try:
            parsed = float(text)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be positive.") from exc
    else:
        raise ValueError(f"{field_name} must be positive.")
    if parsed <= 0.0 or isnan(parsed):
        raise ValueError(f"{field_name} must be positive.")
    return parsed


def _as_positive_int(value: object, *, field_name: str) -> int:
    parsed = _as_positive_float(value, field_name=field_name)
    if not parsed.is_integer():
        raise ValueError(f"{field_name} must be a positive integer.")
    return int(parsed)


def _normalize_side(value: object) -> str:
    side = str(value).strip().lower()
    if not side:
        raise ValueError("side is required.")
    if side in {"long", "buy_open_long"} or "long" in side or side.startswith("buy"):
        return "long"
    if side in {"short", "sell_open_short"} or "short" in side or side.startswith("sell"):
        return "short"
    raise ValueError("side must be long or short.")


def _resolve_signal_payload(signal_payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(signal_payload)
    source_signal = payload.get("source_signal")
    if isinstance(source_signal, dict):
        resolved = dict(source_signal)
        for key in ("side", "event_time"):
            if key in payload and key not in resolved:
                resolved[key] = payload[key]
        return resolved
    return payload


def _validate_geometry(*, side: str, stop_ref: float, entry_ref: float, take_ref: float) -> None:
    if side == "long" and not (stop_ref < entry_ref < take_ref):
        raise ValueError("Invalid long geometry: expected stop_ref < entry_ref < take_ref.")
    if side == "short" and not (take_ref < entry_ref < stop_ref):
        raise ValueError("Invalid short geometry: expected take_ref < entry_ref < stop_ref.")


def simulate_paper_exit_from_bars(
    *,
    fill_payload: dict[str, Any],
    signal_payload: dict[str, Any],
    ohlc_bars: list[dict[str, Any]],
) -> dict[str, Any]:
    fill = dict(fill_payload)
    signal = _resolve_signal_payload(signal_payload)

    side = _normalize_side(signal.get("side", fill.get("side")))
    entry_ref = _as_positive_float(signal.get("entry_ref"), field_name="entry_ref")
    stop_ref = _as_positive_float(signal.get("stop_ref"), field_name="stop_ref")
    take_ref = _as_positive_float(signal.get("take_ref"), field_name="take_ref")
    ttl_sec = _as_positive_int(signal.get("ttl_sec"), field_name="ttl_sec")
    signal_time = _as_datetime_utc(signal.get("event_time"), field_name="event_time")
    entry_time = _as_datetime_utc(
        fill.get("entry_time", fill.get("filled_at")),
        field_name="entry_time",
    )

    _validate_geometry(
        side=side,
        stop_ref=stop_ref,
        entry_ref=entry_ref,
        take_ref=take_ref,
    )

    ttl_boundary = signal_time + timedelta(seconds=ttl_sec)

    sorted_bars: list[dict[str, Any]] = sorted(
        (dict(bar) for bar in ohlc_bars if isinstance(bar, dict)),
        key=lambda row: _as_datetime_utc(row.get("event_time"), field_name="bar.event_time"),
    )

    bars_checked = 0
    for bar in sorted_bars:
        bar_time = _as_datetime_utc(bar.get("event_time"), field_name="bar.event_time")
        if bar_time < entry_time:
            continue

        bars_checked += 1
        high = _as_positive_float(bar.get("high"), field_name="bar.high")
        low = _as_positive_float(bar.get("low"), field_name="bar.low")
        close = _as_positive_float(bar.get("close"), field_name="bar.close")

        if bar_time < ttl_boundary:
            if side == "long":
                stop_hit = low <= stop_ref
                take_hit = high >= take_ref
            else:
                stop_hit = high >= stop_ref
                take_hit = low <= take_ref

            # Conservative policy: in same-bar conflict stop-loss is assumed first.
            if stop_hit:
                return {
                    "exit_price": float(stop_ref),
                    "exit_time": bar_time.isoformat(),
                    "exit_reason": "stop_loss",
                    "source": "historical_bars_exit_policy",
                    "bars_checked": bars_checked,
                    "simulation_only": True,
                }
            if take_hit:
                return {
                    "exit_price": float(take_ref),
                    "exit_time": bar_time.isoformat(),
                    "exit_reason": "take_profit",
                    "source": "historical_bars_exit_policy",
                    "bars_checked": bars_checked,
                    "simulation_only": True,
                }
            continue

        return {
            "exit_price": float(close),
            "exit_time": bar_time.isoformat(),
            "exit_reason": "ttl_exit",
            "source": "historical_bars_exit_policy",
            "bars_checked": bars_checked,
            "simulation_only": True,
        }

    return {
        "exit_price": float(entry_ref),
        "exit_time": entry_time.isoformat(),
        "exit_reason": "no_exit_data",
        "source": "historical_bars_exit_policy",
        "bars_checked": bars_checked,
        "simulation_only": True,
    }
