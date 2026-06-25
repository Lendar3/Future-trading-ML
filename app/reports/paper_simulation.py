from __future__ import annotations

from collections import Counter
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


def _slice_latest(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return [dict(row) for row in rows[:limit]]


def build_paper_simulation_audit_report(
    *,
    paper_intents: list[dict[str, Any]] | None,
    paper_orders: list[dict[str, Any]] | None,
    paper_fills: list[dict[str, Any]] | None,
    paper_intent_executions: list[dict[str, Any]] | None,
    portfolio_snapshot: dict[str, Any] | None,
    limit: int = 50,
) -> dict[str, Any]:
    intents = list(paper_intents or [])
    orders = list(paper_orders or [])
    fills = list(paper_fills or [])
    executions = list(paper_intent_executions or [])
    portfolio = dict(portfolio_snapshot or {})

    warnings: list[str] = [
        "Manual simulation only.",
        "Real execution disabled.",
        "Auto-loop not enabled.",
    ]

    intent_status_counts = Counter(
        str(intent.get("status", "unknown")).strip().lower() or "unknown" for intent in intents
    )

    blocking_reason_counts: Counter[str] = Counter()
    for intent in intents:
        raw_reasons = intent.get("blocking_reasons")
        if isinstance(raw_reasons, list):
            for reason in raw_reasons:
                reason_text = str(reason).strip()
                if reason_text:
                    blocking_reason_counts[reason_text] += 1

    simulated_intent_ids: set[str] = set()
    execution_intent_ids: list[str] = []
    for row in executions:
        intent_id = str(row.get("intent_id", "")).strip()
        if intent_id:
            execution_intent_ids.append(intent_id)
            simulated_intent_ids.add(intent_id)

    candidate_intents = [
        intent for intent in intents if str(intent.get("status", "")).strip().lower() == "candidate"
    ]
    candidate_intent_ids = {
        str(intent.get("intent_id", "")).strip()
        for intent in candidate_intents
        if intent.get("intent_id")
    }
    unsimulated_candidate_intents = len(candidate_intent_ids - simulated_intent_ids)

    replay_count_available = False
    duplicate_replay_count: int | None = None
    if execution_intent_ids:
        duplicates = len(execution_intent_ids) - len(set(execution_intent_ids))
        if duplicates > 0:
            replay_count_available = True
            duplicate_replay_count = duplicates
    if not replay_count_available:
        warnings.append(
            "Duplicate replay count unavailable: "
            "current execution table stores one record per intent."
        )

    long_intents = 0
    short_intents = 0
    no_trade_or_invalid_intents = 0
    unique_symbols: set[str] = set()

    for intent in intents:
        side = str(intent.get("side", "")).strip().lower()
        symbol = str(intent.get("symbol", "")).strip()
        if symbol:
            unique_symbols.add(symbol)
        if side == "long":
            long_intents += 1
        elif side == "short":
            short_intents += 1
        else:
            no_trade_or_invalid_intents += 1

    for order in orders:
        symbol = str(order.get("symbol", "")).strip()
        if symbol:
            unique_symbols.add(symbol)

    for fill in fills:
        symbol = str(fill.get("symbol", "")).strip()
        if symbol:
            unique_symbols.add(symbol)

    portfolio_equity = _as_float_or_none(portfolio.get("equity"))
    portfolio_realized_pnl = _as_float_or_none(
        portfolio.get("realized_pnl_total", portfolio.get("realized_pnl"))
    )
    portfolio_unrealized_pnl = _as_float_or_none(
        portfolio.get("unrealized_pnl_total", portfolio.get("unrealized_pnl"))
    )
    portfolio_fees_paid = _as_float_or_none(
        portfolio.get("fees_paid_total", portfolio.get("fees_paid"))
    )

    if portfolio_realized_pnl is None:
        warnings.append("Portfolio realized PnL is unavailable.")
    if portfolio_unrealized_pnl is None:
        warnings.append("Portfolio unrealized PnL is unavailable.")
    if portfolio_fees_paid is None:
        warnings.append("Portfolio fees-paid total is unavailable.")

    summary: dict[str, Any] = {
        "total_intents": len(intents),
        "blocked_intents": intent_status_counts.get("blocked", 0),
        "candidate_intents": intent_status_counts.get("candidate", 0),
        "simulated_intents": len(simulated_intent_ids),
        "unsimulated_candidate_intents": unsimulated_candidate_intents,
        "duplicate_replay_count": duplicate_replay_count,
        "total_orders": len(orders),
        "total_fills": len(fills),
        "unique_symbols": sorted(unique_symbols),
        "long_intents": long_intents,
        "short_intents": short_intents,
        "no_trade_or_invalid_intents": no_trade_or_invalid_intents,
        "paper_equity": portfolio_equity,
        "paper_realized_pnl": portfolio_realized_pnl,
        "paper_unrealized_pnl": portfolio_unrealized_pnl,
        "paper_fees_paid": portfolio_fees_paid,
    }

    simulation_status_counts = {
        "execution_records": len(executions),
        "simulated_intents": len(simulated_intent_ids),
        "candidate_intents": intent_status_counts.get("candidate", 0),
        "unsimulated_candidate_intents": unsimulated_candidate_intents,
    }

    # Compact latest simulation rows with intent metadata to simplify investor-facing UI.
    intent_index = {
        str(intent.get("intent_id", "")).strip(): intent
        for intent in intents
        if str(intent.get("intent_id", "")).strip()
    }
    latest_simulations: list[dict[str, Any]] = []
    for row in executions:
        item = dict(row)
        intent_id = str(row.get("intent_id", "")).strip()
        intent = intent_index.get(intent_id, {})
        if intent:
            item["symbol"] = intent.get("symbol")
            item["side"] = intent.get("side")
            item["intent_status"] = intent.get("status")
        latest_simulations.append(item)

    positions = portfolio.get("positions")
    positions_payload = positions if isinstance(positions, dict) else {}
    portfolio_summary = {
        "snapshot_id": portfolio.get("snapshot_id"),
        "event_time": portfolio.get("event_time"),
        "equity": portfolio_equity,
        "margin_used": _as_float_or_none(portfolio.get("margin_used")),
        "available_margin": _as_float_or_none(portfolio.get("available_margin")),
        "positions_count": len(positions_payload),
        "positions": positions_payload,
        "realized_pnl": portfolio_realized_pnl,
        "unrealized_pnl": portfolio_unrealized_pnl,
        "fees_paid": portfolio_fees_paid,
    }

    return {
        "created_at": _utc_now_iso(),
        "summary": summary,
        "intent_status_counts": dict(intent_status_counts),
        "simulation_status_counts": simulation_status_counts,
        "blocking_reason_counts": dict(blocking_reason_counts),
        "latest_intents": _slice_latest(intents, limit),
        "latest_simulations": _slice_latest(latest_simulations, limit),
        "latest_orders": _slice_latest(orders, limit),
        "latest_fills": _slice_latest(fills, limit),
        "portfolio_summary": portfolio_summary,
        "warnings": warnings,
    }
