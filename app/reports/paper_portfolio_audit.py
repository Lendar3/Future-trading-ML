from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


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


def _parse_iso_datetime(value: object) -> datetime | None:
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


def _snapshot_id_for_close(close_id: str) -> str:
    digest = hashlib.sha256(close_id.encode()).hexdigest()[:24]
    return f"psnap_close_{digest}"


def build_paper_portfolio_audit_report(
    *,
    paper_fills: list[dict[str, Any]] | None,
    paper_trade_closes: list[dict[str, Any]] | None,
    portfolio_snapshots: list[dict[str, Any]] | None,
    current_portfolio_snapshot: dict[str, Any] | None,
    limit: int = 500,
) -> dict[str, Any]:
    fills = list(paper_fills or [])
    closes = list(paper_trade_closes or [])
    snapshots = list(portfolio_snapshots or [])
    current_portfolio = dict(current_portfolio_snapshot or {})

    warnings: list[str] = [
        "Manual simulation only.",
        "Real execution disabled.",
        "Auto-loop not enabled.",
    ]
    position_consistency_warnings: list[str] = []

    if not snapshots and current_portfolio:
        snapshots = [current_portfolio]

    normalized_snapshots: list[dict[str, Any]] = []
    for snapshot in snapshots:
        dt = _parse_iso_datetime(snapshot.get("event_time"))
        normalized_snapshots.append(
            {
                "snapshot_id": snapshot.get("snapshot_id"),
                "event_time": dt.isoformat() if dt else snapshot.get("event_time"),
                "equity": _as_float_or_none(snapshot.get("equity")),
                "margin_used": _as_float_or_none(snapshot.get("margin_used")),
                "available_margin": _as_float_or_none(snapshot.get("available_margin")),
                "positions": (
                    dict(snapshot.get("positions", {}))
                    if isinstance(snapshot.get("positions"), dict)
                    else {}
                ),
            }
        )

    normalized_snapshots = sorted(
        normalized_snapshots,
        key=lambda row: (
            _parse_iso_datetime(row.get("event_time")) or datetime.min.replace(tzinfo=UTC),
            str(row.get("snapshot_id", "")),
        ),
    )

    latest_snapshot = normalized_snapshots[-1] if normalized_snapshots else {}
    latest_positions = latest_snapshot.get("positions", {})
    if not isinstance(latest_positions, dict):
        latest_positions = {}

    open_positions: list[dict[str, Any]] = []
    position_entry_fill_ids: set[str] = set()
    for symbol, raw_position in latest_positions.items():
        if not isinstance(raw_position, dict):
            continue
        status = str(raw_position.get("status", "open")).strip().lower() or "open"
        if status != "open":
            continue
        row = dict(raw_position)
        row["symbol"] = str(row.get("symbol", symbol)).strip() or str(symbol)
        open_positions.append(row)
        entry_fill_id = str(row.get("entry_fill_id", "")).strip()
        if entry_fill_id:
            position_entry_fill_ids.add(entry_fill_id)

    open_fills = [
        fill for fill in fills if str(fill.get("pnl_status", "")).strip().lower() == "open"
    ]
    open_fill_count = len(open_fills)
    closed_trade_count = len(closes)

    fill_ids = {str(fill.get("fill_id", "")).strip() for fill in fills if fill.get("fill_id")}
    orphan_fills: list[dict[str, Any]] = []
    for fill in open_fills:
        fill_id = str(fill.get("fill_id", "")).strip()
        if not fill_id:
            continue
        if fill_id not in position_entry_fill_ids:
            orphan_fills.append(
                {
                    "fill_id": fill_id,
                    "symbol": fill.get("symbol"),
                    "side": fill.get("side"),
                    "reason": "Open fill has no matching open position.",
                }
            )
    if orphan_fills:
        position_consistency_warnings.append("Open fills without matching positions detected.")

    orphan_closes: list[dict[str, Any]] = []
    for close in closes:
        entry_fill_id = str(close.get("entry_fill_id", "")).strip()
        if not entry_fill_id:
            continue
        if entry_fill_id not in fill_ids:
            orphan_closes.append(
                {
                    "close_id": close.get("close_id"),
                    "entry_fill_id": entry_fill_id,
                    "symbol": close.get("symbol"),
                    "reason": "Close references missing fill.",
                }
            )
    if orphan_closes:
        position_consistency_warnings.append("Closes referencing missing fills detected.")

    for row in open_positions:
        side = str(row.get("side", "")).strip().lower()
        if side not in {"long", "short"}:
            position_consistency_warnings.append(
                f"Position {row.get('symbol')} has invalid side: {row.get('side')!r}."
            )

    snapshots_by_id = {
        str(snapshot.get("snapshot_id", "")): snapshot
        for snapshot in normalized_snapshots
        if str(snapshot.get("snapshot_id", ""))
    }
    equity_checks: list[dict[str, Any]] = []
    for close in closes:
        close_id = str(close.get("close_id", "")).strip()
        if not close_id:
            continue
        snapshot_id = _snapshot_id_for_close(close_id)
        close_snapshot = snapshots_by_id.get(snapshot_id)
        if close_snapshot is None:
            equity_checks.append(
                {
                    "close_id": close_id,
                    "snapshot_id": snapshot_id,
                    "is_consistent": None,
                    "reason": "Close snapshot not found.",
                }
            )
            continue
        realized_pnl = _as_float_or_none(close.get("realized_pnl"))
        actual_equity = _as_float_or_none(close_snapshot.get("equity"))
        if realized_pnl is None or actual_equity is None:
            equity_checks.append(
                {
                    "close_id": close_id,
                    "snapshot_id": snapshot_id,
                    "is_consistent": None,
                    "reason": "Missing realized_pnl or snapshot equity.",
                }
            )
            continue

        close_dt = _parse_iso_datetime(close_snapshot.get("event_time"))
        previous_candidates = [
            snap
            for snap in normalized_snapshots
            if str(snap.get("snapshot_id")) != snapshot_id
            and (
                (snap_dt := _parse_iso_datetime(snap.get("event_time"))) is not None
                and close_dt is not None
                and snap_dt <= close_dt
            )
        ]
        if not previous_candidates:
            equity_checks.append(
                {
                    "close_id": close_id,
                    "snapshot_id": snapshot_id,
                    "is_consistent": None,
                    "reason": "Previous snapshot not found.",
                }
            )
            continue
        prev = previous_candidates[-1]
        prev_equity = _as_float_or_none(prev.get("equity"))
        if prev_equity is None:
            equity_checks.append(
                {
                    "close_id": close_id,
                    "snapshot_id": snapshot_id,
                    "is_consistent": None,
                    "reason": "Previous equity missing.",
                }
            )
            continue
        expected_equity = prev_equity + realized_pnl
        is_consistent = abs(actual_equity - expected_equity) <= 1e-9
        equity_checks.append(
            {
                "close_id": close_id,
                "snapshot_id": snapshot_id,
                "previous_snapshot_id": prev.get("snapshot_id"),
                "previous_equity": prev_equity,
                "realized_pnl": realized_pnl,
                "expected_equity": expected_equity,
                "actual_equity": actual_equity,
                "is_consistent": is_consistent,
            }
        )
        if not is_consistent:
            position_consistency_warnings.append(
                "Equity mismatch for close "
                f"{close_id}: expected {expected_equity}, got {actual_equity}."
            )

    consistency_values = [
        row.get("is_consistent")
        for row in equity_checks
        if isinstance(row.get("is_consistent"), bool)
    ]
    all_consistent: bool | None = None
    if consistency_values:
        all_consistent = all(bool(value) for value in consistency_values)

    equity_consistency = {
        "all_consistent": all_consistent,
        "checked_closes": len(consistency_values),
        "checks": equity_checks[:limit],
    }

    summary = {
        "total_fills": len(fills),
        "total_closes": len(closes),
        "total_snapshots": len(normalized_snapshots),
        "open_positions_count": len(open_positions),
        "open_fill_count": open_fill_count,
        "orphan_fill_count": len(orphan_fills),
        "orphan_close_count": len(orphan_closes),
        "equity_consistency_all": all_consistent,
    }

    return {
        "created_at": _utc_now_iso(),
        "summary": summary,
        "open_positions": open_positions[:limit],
        "closed_trade_count": closed_trade_count,
        "open_fill_count": open_fill_count,
        "orphan_fills": orphan_fills[:limit],
        "orphan_closes": orphan_closes[:limit],
        "position_consistency_warnings": position_consistency_warnings,
        "equity_consistency": equity_consistency,
        "warnings": warnings,
    }
