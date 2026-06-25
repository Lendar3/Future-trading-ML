from __future__ import annotations

from collections import defaultdict
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


def _fill_realized_pnl(fill: dict[str, Any]) -> float | None:
    for key in ("realized_pnl", "pnl", "trade_pnl", "net_pnl"):
        pnl = _as_float_or_none(fill.get(key))
        if pnl is not None:
            return pnl
    return None


def _closed_trade_attribution_time(fill: dict[str, Any]) -> tuple[datetime | None, bool]:
    status = str(fill.get("pnl_status", "")).strip().lower()
    if status != "closed":
        return None, False
    if _fill_realized_pnl(fill) is None:
        return None, False

    exit_time = _parse_iso_datetime(fill.get("exit_time"))
    if exit_time is not None:
        return exit_time, False

    closed_at = _parse_iso_datetime(fill.get("closed_at"))
    if closed_at is not None:
        return closed_at, False

    filled_at = _parse_iso_datetime(fill.get("filled_at"))
    if filled_at is not None:
        return filled_at, True

    return None, False


def _max_drawdown_from_equity_curve(
    curve: list[dict[str, Any]],
) -> tuple[float | None, float | None]:
    if not curve:
        return None, None

    peak: float | None = None
    max_dd = 0.0
    max_dd_pct = 0.0

    for point in curve:
        equity = _as_float_or_none(point.get("equity"))
        if equity is None:
            continue
        if peak is None or equity > peak:
            peak = equity
            continue
        drawdown = peak - equity
        if drawdown > max_dd:
            max_dd = drawdown
        if peak > 0.0:
            dd_pct = drawdown / peak
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

    if peak is None:
        return None, None
    return max_dd, max_dd_pct


def build_paper_track_record_report(
    *,
    paper_orders: list[dict[str, Any]] | None,
    paper_fills: list[dict[str, Any]] | None,
    portfolio_snapshots: list[dict[str, Any]] | None,
    current_portfolio_snapshot: dict[str, Any] | None,
    limit: int = 500,
) -> dict[str, Any]:
    orders = list(paper_orders or [])
    fills = list(paper_fills or [])
    snapshots = list(portfolio_snapshots or [])
    current_portfolio = dict(current_portfolio_snapshot or {})

    warnings: list[str] = [
        "Manual simulation only.",
        "Real execution disabled.",
        "Auto-loop not enabled.",
    ]

    orders_by_id = {
        str(order.get("order_id", "")).strip(): dict(order)
        for order in orders
        if str(order.get("order_id", "")).strip()
    }

    unique_symbols = {
        str(item.get("symbol", "")).strip()
        for item in [*orders, *fills]
        if str(item.get("symbol", "")).strip()
    }

    first_fill_time: str | None = None
    last_fill_time: str | None = None
    fill_times: list[datetime] = []

    for fill in fills:
        dt = _parse_iso_datetime(fill.get("filled_at"))
        if dt is None:
            continue
        fill_times.append(dt)

    if fill_times:
        first_fill_time = min(fill_times).isoformat()
        last_fill_time = max(fill_times).isoformat()

    normalized_snapshots: list[dict[str, Any]] = []
    for snapshot in snapshots:
        event_dt = _parse_iso_datetime(snapshot.get("event_time"))
        normalized_snapshots.append(
            {
                "snapshot_id": snapshot.get("snapshot_id"),
                "event_time": event_dt.isoformat() if event_dt else snapshot.get("event_time"),
                "equity": _as_float_or_none(snapshot.get("equity")),
                "margin_used": _as_float_or_none(snapshot.get("margin_used")),
                "available_margin": _as_float_or_none(snapshot.get("available_margin")),
            }
        )

    normalized_snapshots = sorted(
        normalized_snapshots,
        key=lambda item: _parse_iso_datetime(item.get("event_time"))
        or datetime.min.replace(tzinfo=UTC),
    )

    if not normalized_snapshots and current_portfolio:
        current_dt = _parse_iso_datetime(current_portfolio.get("event_time"))
        normalized_snapshots = [
            {
                "snapshot_id": current_portfolio.get("snapshot_id"),
                "event_time": (
                    current_dt.isoformat()
                    if current_dt is not None
                    else current_portfolio.get("event_time")
                ),
                "equity": _as_float_or_none(current_portfolio.get("equity")),
                "margin_used": _as_float_or_none(current_portfolio.get("margin_used")),
                "available_margin": _as_float_or_none(current_portfolio.get("available_margin")),
            }
        ]

    equity_curve = normalized_snapshots[-limit:]

    starting_equity = None
    if equity_curve:
        starting_equity = _as_float_or_none(equity_curve[0].get("equity"))

    paper_equity = _as_float_or_none(current_portfolio.get("equity"))
    if paper_equity is None and equity_curve:
        paper_equity = _as_float_or_none(equity_curve[-1].get("equity"))

    total_realized_pnl = _as_float_or_none(
        current_portfolio.get("realized_pnl_total", current_portfolio.get("realized_pnl"))
    )
    total_unrealized_pnl = _as_float_or_none(
        current_portfolio.get("unrealized_pnl_total", current_portfolio.get("unrealized_pnl"))
    )
    total_fees_paid = _as_float_or_none(
        current_portfolio.get("fees_paid_total", current_portfolio.get("fees_paid"))
    )
    if total_fees_paid is None:
        computed_fee = sum(
            fee
            for fee in (_as_float_or_none(fill.get("fees_paid", fill.get("fee"))) for fill in fills)
            if fee is not None
        )
        if computed_fee > 0.0:
            total_fees_paid = computed_fee

    pnl_status_counts: dict[str, int] = defaultdict(int)
    open_trades = 0
    closed_trades = 0
    total_notional = 0.0

    for fill in fills:
        status = str(fill.get("pnl_status", "unavailable")).strip().lower() or "unavailable"
        pnl_status_counts[status] += 1
        if status == "open":
            open_trades += 1
        elif status == "closed":
            closed_trades += 1
        notional = _as_float_or_none(fill.get("notional"))
        if notional is None:
            qty = _as_float_or_none(fill.get("quantity"))
            price = _as_float_or_none(fill.get("price"))
            if qty is not None and price is not None:
                notional = abs(qty * price)
        if notional is not None:
            total_notional += notional

    closed_realized_pnl = [
        p
        for p in (
            (
                _fill_realized_pnl(fill)
                if str(fill.get("pnl_status", "")).strip().lower() == "closed"
                else None
            )
            for fill in fills
        )
        if p is not None
    ]
    pnl_available = len(closed_realized_pnl) > 0

    if not pnl_available:
        warnings.append("No closed paper trades with realized PnL yet.")
    if closed_trades > 0 and len(normalized_snapshots) <= 1:
        warnings.append("Portfolio equity is not automatically updated by close simulation yet.")
    if total_realized_pnl is None and pnl_available:
        total_realized_pnl = float(sum(closed_realized_pnl))

    total_pnl = sum(closed_realized_pnl) if pnl_available else None
    win_values = [p for p in closed_realized_pnl if p > 0.0]
    loss_values = [p for p in closed_realized_pnl if p < 0.0]
    win_count = len(win_values) if pnl_available else None
    loss_count = len(loss_values) if pnl_available else None
    win_rate = None
    if pnl_available and len(closed_realized_pnl) > 0:
        win_rate = len(win_values) / len(closed_realized_pnl)

    avg_win = (sum(win_values) / len(win_values)) if win_values else None
    avg_loss = (sum(loss_values) / len(loss_values)) if loss_values else None

    avg_win_loss_ratio = None
    if avg_win is not None and avg_loss is not None and avg_loss != 0.0:
        avg_win_loss_ratio = abs(avg_win / avg_loss)

    gross_profit = sum(win_values)
    gross_loss_abs = abs(sum(loss_values))
    profit_factor = None
    if pnl_available and gross_loss_abs > 0.0:
        profit_factor = gross_profit / gross_loss_abs

    total_return_pct = None
    if starting_equity is not None and paper_equity is not None and starting_equity > 0.0:
        total_return_pct = (paper_equity - starting_equity) / starting_equity

    if starting_equity is None:
        warnings.append("Starting equity is unavailable.")

    max_drawdown, max_drawdown_pct = _max_drawdown_from_equity_curve(equity_curve)
    if max_drawdown is None:
        warnings.append("Max drawdown is unavailable due to missing equity curve data.")

    daily_aggregates: dict[str, dict[str, Any]] = defaultdict(lambda: {"fill_count": 0, "pnl": 0.0})
    monthly_aggregates: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"fill_count": 0, "pnl": 0.0}
    )
    close_time_fallback_used = False

    for fill in fills:
        fill_dt = _parse_iso_datetime(fill.get("filled_at"))
        if fill_dt is None:
            continue
        day_key = fill_dt.date().isoformat()
        month_key = f"{fill_dt.year:04d}-{fill_dt.month:02d}"

        daily_aggregates[day_key]["fill_count"] += 1
        monthly_aggregates[month_key]["fill_count"] += 1

        pnl = _fill_realized_pnl(fill)
        if pnl is None:
            continue
        attribution_dt, used_fallback = _closed_trade_attribution_time(fill)
        if attribution_dt is None:
            continue
        if used_fallback:
            close_time_fallback_used = True
        realized_day_key = attribution_dt.date().isoformat()
        realized_month_key = f"{attribution_dt.year:04d}-{attribution_dt.month:02d}"
        daily_aggregates[realized_day_key]["pnl"] += pnl
        monthly_aggregates[realized_month_key]["pnl"] += pnl

    if close_time_fallback_used:
        warnings.append(
            "Some closed trades are missing exit_time/closed_at; fallback attribution was used."
        )

    daily_summary: list[dict[str, Any]] = []
    for day in sorted(daily_aggregates.keys()):
        row = daily_aggregates[day]
        daily_summary.append(
            {
                "date": day,
                "fill_count": int(row["fill_count"]),
                "pnl": (float(row["pnl"]) if isinstance(row["pnl"], (int, float)) else None),
            }
        )

    monthly_summary: list[dict[str, Any]] = []
    for month in sorted(monthly_aggregates.keys()):
        row = monthly_aggregates[month]
        monthly_summary.append(
            {
                "month": month,
                "fill_count": int(row["fill_count"]),
                "pnl": (float(row["pnl"]) if isinstance(row["pnl"], (int, float)) else None),
            }
        )

    day_pnl_values = [
        row["pnl"] for row in daily_summary if isinstance(row.get("pnl"), (int, float))
    ]
    best_day_pnl = max(day_pnl_values) if day_pnl_values else None
    worst_day_pnl = min(day_pnl_values) if day_pnl_values else None
    positive_day_count = sum(1 for pnl in day_pnl_values if pnl > 0.0) if day_pnl_values else None
    negative_day_count = sum(1 for pnl in day_pnl_values if pnl < 0.0) if day_pnl_values else None

    latest_trades: list[dict[str, Any]] = []
    for fill in fills[:limit]:
        order_id = str(fill.get("order_id", "")).strip()
        order = orders_by_id.get(order_id, {})
        latest_trades.append(
            {
                "fill_id": fill.get("fill_id"),
                "order_id": fill.get("order_id"),
                "symbol": fill.get("symbol"),
                "side": fill.get("side"),
                "quantity": _as_float_or_none(fill.get("quantity")),
                "price": _as_float_or_none(fill.get("price")),
                "fee": _as_float_or_none(fill.get("fee")),
                "fees_paid": _as_float_or_none(fill.get("fees_paid", fill.get("fee"))),
                "filled_at": fill.get("filled_at"),
                "entry_price": _as_float_or_none(fill.get("entry_price", fill.get("price"))),
                "entry_time": fill.get("entry_time", fill.get("filled_at")),
                "notional": _as_float_or_none(fill.get("notional")),
                "pnl_status": fill.get("pnl_status"),
                "realized_pnl": _as_float_or_none(fill.get("realized_pnl")),
                "unrealized_pnl": _as_float_or_none(fill.get("unrealized_pnl")),
                "exit_price": _as_float_or_none(fill.get("exit_price")),
                "exit_time": fill.get("exit_time"),
                "close_reason": fill.get("close_reason"),
                "status": order.get("status"),
                "order_created_at": order.get("created_at"),
                "pnl": _fill_realized_pnl(fill),
            }
        )

    summary = {
        "total_orders": len(orders),
        "total_fills": len(fills),
        "filled_trades": len(fills),
        "unique_symbols": sorted(unique_symbols),
        "first_fill_time": first_fill_time,
        "last_fill_time": last_fill_time,
        "paper_equity": paper_equity,
        "starting_equity": starting_equity,
        "total_realized_pnl": total_realized_pnl,
        "total_unrealized_pnl": total_unrealized_pnl,
        "total_fees_paid": total_fees_paid,
        "open_trades": open_trades,
        "closed_trades": closed_trades,
        "pnl_status_counts": dict(pnl_status_counts),
        "total_notional": total_notional,
    }

    performance_metrics = {
        "total_pnl": total_pnl,
        "total_return_pct": total_return_pct,
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_win_loss_ratio": avg_win_loss_ratio,
        "profit_factor": profit_factor,
    }

    risk_metrics = {
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "best_day_pnl": best_day_pnl,
        "worst_day_pnl": worst_day_pnl,
        "positive_day_count": positive_day_count,
        "negative_day_count": negative_day_count,
    }

    return {
        "created_at": _utc_now_iso(),
        "summary": summary,
        "performance_metrics": performance_metrics,
        "risk_metrics": risk_metrics,
        "daily_summary": daily_summary,
        "monthly_summary": monthly_summary,
        "latest_trades": latest_trades,
        "equity_curve": equity_curve,
        "warnings": warnings,
    }
