from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol


class PaperServiceRepositoryProtocol(Protocol):
    # Read methods used by paper services
    def get_paper_intent(self, intent_id: str) -> dict[str, Any] | None: ...

    def get_paper_intent_execution(self, intent_id: str) -> dict[str, Any] | None: ...

    def get_paper_order(self, order_id: str) -> dict[str, Any] | None: ...

    def get_paper_fill(self, fill_id: str) -> dict[str, Any] | None: ...

    def get_paper_trade_close_by_fill(self, fill_id: str) -> dict[str, Any] | None: ...

    def get_portfolio_snapshot_for_fill(self, fill_id: str) -> dict[str, Any] | None: ...

    def get_portfolio_snapshot_for_close(self, close_id: str) -> dict[str, Any] | None: ...

    def latest_portfolio(self) -> dict[str, Any]: ...

    # Write methods used by paper services
    def upsert_paper_intent(
        self,
        *,
        intent_id: str,
        model_run_id: str,
        signal_id: str,
        symbol: str,
        side: str,
        event_time: datetime,
        status: str,
        approved_for_paper_demo: bool,
        created_at: datetime,
        intent_json: dict[str, Any],
    ) -> None: ...

    def create_order_and_fill(
        self,
        order_id: str,
        fill_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        fee: float,
        created_at: datetime,
    ) -> None: ...

    def upsert_paper_intent_execution(
        self,
        *,
        intent_id: str,
        order_id: str,
        fill_id: str,
        created_at: datetime,
    ) -> None: ...

    def create_paper_trade_close(self, close_payload: dict[str, Any]) -> dict[str, Any]: ...

    def create_portfolio_snapshot_after_fill(
        self, fill_payload: dict[str, Any]
    ) -> dict[str, Any]: ...

    def create_portfolio_snapshot_after_close(
        self, close_payload: dict[str, Any]
    ) -> dict[str, Any]: ...

    def append_paper_lifecycle_event(
        self,
        *,
        event_type: str,
        entity_type: str,
        entity_id: str,
        related_entity_id: str | None = None,
        created_at: datetime | None = None,
        event_json: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...
