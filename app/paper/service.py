from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.paper.accounting import build_paper_close_accounting
from app.paper.order_requests import build_paper_order_request_from_intent
from app.paper.repository_protocols import PaperServiceRepositoryProtocol


class PaperServiceError(Exception):
    def __init__(self, *, status_code: int, detail: str | dict[str, Any]) -> None:
        super().__init__(str(detail))
        self.status_code = status_code
        self.detail = detail


def _parse_iso_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    raise ValueError("Expected datetime or ISO datetime string.")


def close_paper_fill_service(
    *,
    repo: PaperServiceRepositoryProtocol,
    fill_id: str,
    exit_price: float,
    exit_fee: float,
    close_reason: str,
    closed_at: datetime,
) -> dict[str, object]:
    fill = repo.get_paper_fill(fill_id)
    if fill is None:
        raise PaperServiceError(status_code=404, detail="Paper fill not found.")

    existing_close = repo.get_paper_trade_close_by_fill(fill_id)
    if existing_close is not None:
        close_id = str(existing_close.get("close_id", "")).strip()
        portfolio_snapshot = repo.get_portfolio_snapshot_for_close(close_id) if close_id else None
        repo.append_paper_lifecycle_event(
            event_type="paper_close_replayed",
            entity_type="paper_fill",
            entity_id=fill_id,
            related_entity_id=close_id or None,
            event_json={
                "fill_id": fill_id,
                "close_id": close_id or None,
            },
        )
        return {
            "simulation_only": True,
            "manual_trigger": True,
            "real_execution": False,
            "close": existing_close,
            "portfolio_snapshot": portfolio_snapshot or repo.latest_portfolio(),
        }

    try:
        accounting = build_paper_close_accounting(
            entry_fill_payload=fill,
            exit_price=exit_price,
            exit_time=closed_at,
            exit_fee=exit_fee,
            close_reason=close_reason,
        )
    except ValueError as exc:
        raise PaperServiceError(status_code=400, detail=str(exc)) from exc

    close_record = {
        **accounting,
        "order_id": str(fill["order_id"]),
        "symbol": str(fill["symbol"]),
    }
    persisted = repo.create_paper_trade_close(close_record)
    updated_portfolio_snapshot = repo.create_portfolio_snapshot_after_close(persisted)
    close_id = str(persisted.get("close_id", "")).strip()
    repo.append_paper_lifecycle_event(
        event_type="paper_fill_closed",
        entity_type="paper_fill",
        entity_id=fill_id,
        related_entity_id=close_id or None,
        created_at=_parse_iso_datetime(persisted["exit_time"]),
        event_json={
            "fill_id": fill_id,
            "close_id": close_id or None,
            "order_id": str(fill["order_id"]),
            "symbol": str(fill["symbol"]),
            "realized_pnl": persisted.get("realized_pnl"),
        },
    )
    return {
        "simulation_only": True,
        "manual_trigger": True,
        "real_execution": False,
        "close": persisted,
        "portfolio_snapshot": updated_portfolio_snapshot,
    }


def persist_paper_intent_service(
    *,
    repo: PaperServiceRepositoryProtocol,
    intent_payload: dict[str, Any],
) -> dict[str, Any]:
    intent = dict(intent_payload)
    try:
        event_time = _parse_iso_datetime(intent["event_time"])
        created_at = _parse_iso_datetime(intent["created_at"])
    except (KeyError, ValueError) as exc:
        raise PaperServiceError(
            status_code=400,
            detail=f"Invalid intent timestamps: {exc}",
        ) from exc

    repo.upsert_paper_intent(
        intent_id=str(intent["intent_id"]),
        model_run_id=str(intent["model_run_id"]),
        signal_id=str(intent["signal_id"]),
        symbol=str(intent["symbol"]),
        side=str(intent["side"]),
        event_time=event_time,
        status=str(intent["status"]),
        approved_for_paper_demo=bool(intent["approved_for_paper_demo"]),
        created_at=created_at,
        intent_json=intent,
    )
    repo.append_paper_lifecycle_event(
        event_type="intent_created",
        entity_type="paper_intent",
        entity_id=str(intent["intent_id"]),
        created_at=created_at,
        event_json={
            "intent_id": str(intent["intent_id"]),
            "model_run_id": str(intent["model_run_id"]),
            "signal_id": str(intent["signal_id"]),
            "status": str(intent["status"]),
            "approved_for_paper_demo": bool(intent["approved_for_paper_demo"]),
        },
    )
    return intent


def simulate_order_from_intent_service(
    *,
    repo: PaperServiceRepositoryProtocol,
    intent_id: str,
    execution_config: dict[str, Any],
) -> dict[str, object]:
    intent = repo.get_paper_intent(intent_id)
    if intent is None:
        raise PaperServiceError(status_code=404, detail="Paper intent not found.")

    existing_execution = repo.get_paper_intent_execution(intent_id)
    if existing_execution is not None:
        existing_order = repo.get_paper_order(existing_execution["order_id"])
        existing_fill = repo.get_paper_fill(existing_execution["fill_id"])
        if existing_order is None or existing_fill is None:
            raise PaperServiceError(
                status_code=409,
                detail="Intent simulation record exists but order/fill is missing.",
            )
        portfolio_snapshot = repo.get_portfolio_snapshot_for_fill(
            str(existing_execution["fill_id"])
        )
        repo.append_paper_lifecycle_event(
            event_type="intent_simulation_replayed",
            entity_type="paper_intent",
            entity_id=intent_id,
            related_entity_id=str(existing_execution["fill_id"]),
            event_json={
                "intent_id": intent_id,
                "order_id": str(existing_execution["order_id"]),
                "fill_id": str(existing_execution["fill_id"]),
            },
        )
        return {
            "intent_id": intent_id,
            "source_intent_id": intent_id,
            "order": existing_order,
            "fill": existing_fill,
            "portfolio_snapshot": portfolio_snapshot or repo.latest_portfolio(),
            "manual_trigger": True,
            "simulation_only": True,
            "real_execution": False,
        }

    order_request = build_paper_order_request_from_intent(
        intent_payload=intent,
        execution_config=execution_config,
    )
    if order_request["status"] != "ready":
        raise PaperServiceError(
            status_code=409,
            detail={
                "message": "Paper intent is not eligible for manual simulation.",
                "blocking_reasons": order_request["blocking_reasons"],
            },
        )

    try:
        created_at = _parse_iso_datetime(order_request["created_at"])
        quantity = float(order_request["qty"])
        entry_ref = float(order_request["entry_ref"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PaperServiceError(
            status_code=400,
            detail=f"Invalid order request fields: {exc}",
        ) from exc
    fee = abs(quantity * entry_ref) * 0.00055

    order_id = str(order_request["order_id"])
    fill_id = str(order_request["fill_id"])
    repo.create_order_and_fill(
        order_id=order_id,
        fill_id=fill_id,
        symbol=str(order_request["symbol"]),
        side=str(order_request["order_side"]),
        quantity=quantity,
        price=entry_ref,
        fee=fee,
        created_at=created_at,
    )
    repo.upsert_paper_intent_execution(
        intent_id=intent_id,
        order_id=order_id,
        fill_id=fill_id,
        created_at=created_at,
    )

    order = repo.get_paper_order(order_id)
    fill = repo.get_paper_fill(fill_id)
    if order is None or fill is None:
        raise PaperServiceError(
            status_code=500,
            detail="Failed to load simulated paper order/fill after persistence.",
        )
    portfolio_snapshot = repo.create_portfolio_snapshot_after_fill(fill)
    repo.append_paper_lifecycle_event(
        event_type="intent_simulated",
        entity_type="paper_intent",
        entity_id=intent_id,
        related_entity_id=fill_id,
        created_at=created_at,
        event_json={
            "intent_id": intent_id,
            "order_id": order_id,
            "fill_id": fill_id,
            "symbol": str(order_request["symbol"]),
            "side": str(order_request["order_side"]),
        },
    )
    return {
        "intent_id": intent_id,
        "source_intent_id": intent_id,
        "order": order,
        "fill": fill,
        "portfolio_snapshot": portfolio_snapshot,
        "manual_trigger": True,
        "simulation_only": True,
        "real_execution": False,
    }
