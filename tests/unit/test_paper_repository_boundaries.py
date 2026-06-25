from datetime import UTC
from typing import Any

from app.paper.repository_protocols import PaperServiceRepositoryProtocol
from app.storage.paper_read_repository import PaperReadRepository
from app.storage.paper_write_repository import PaperWriteRepository
from app.storage.repositories import PostgresPaperRepository

READ_METHODS = {
    "latest_paper_intents",
    "get_paper_intent",
    "latest_paper_orders",
    "get_paper_order",
    "latest_fills",
    "get_paper_fill",
    "latest_paper_trade_closes",
    "get_paper_trade_close_by_fill",
    "latest_paper_intent_executions",
    "get_paper_intent_execution",
    "latest_portfolio",
    "latest_portfolio_snapshots",
    "get_portfolio_snapshot_for_fill",
    "get_portfolio_snapshot_for_close",
    "latest_paper_lifecycle_events",
}

WRITE_METHODS = {
    "reset_demo_account",
    "upsert_paper_intent",
    "upsert_paper_intent_execution",
    "create_order_and_fill",
    "create_paper_trade_close",
    "create_portfolio_snapshot_after_fill",
    "create_portfolio_snapshot_after_close",
    "append_paper_lifecycle_event",
}


def test_postgres_paper_repository_keeps_read_and_write_interface() -> None:
    assert issubclass(PaperWriteRepository, PaperReadRepository)
    assert issubclass(PostgresPaperRepository, PaperWriteRepository)

    for method_name in READ_METHODS | WRITE_METHODS:
        assert hasattr(PostgresPaperRepository, method_name)


def test_paper_read_repository_exposes_only_read_contract() -> None:
    for method_name in READ_METHODS:
        assert hasattr(PaperReadRepository, method_name)
    for method_name in WRITE_METHODS:
        assert not hasattr(PaperReadRepository, method_name)


def test_paper_write_repository_exposes_read_and_write_contracts() -> None:
    for method_name in READ_METHODS | WRITE_METHODS:
        assert hasattr(PaperWriteRepository, method_name)


def test_postgres_paper_repository_satisfies_service_protocol() -> None:
    protocol_methods = {
        "get_paper_intent",
        "get_paper_intent_execution",
        "get_paper_order",
        "get_paper_fill",
        "get_paper_trade_close_by_fill",
        "get_portfolio_snapshot_for_fill",
        "get_portfolio_snapshot_for_close",
        "latest_portfolio",
        "upsert_paper_intent",
        "create_order_and_fill",
        "upsert_paper_intent_execution",
        "create_paper_trade_close",
        "create_portfolio_snapshot_after_fill",
        "create_portfolio_snapshot_after_close",
        "append_paper_lifecycle_event",
    }

    for method_name in protocol_methods:
        assert hasattr(PaperServiceRepositoryProtocol, method_name)
        assert hasattr(PostgresPaperRepository, method_name)


class _CaptureSession:
    def __init__(self) -> None:
        self.params: list[dict[str, Any]] = []

    def execute(self, _: object, params: dict[str, Any] | None = None) -> None:
        if params is not None:
            self.params.append(params)


def test_reset_demo_account_uses_explicit_utc_timestamp() -> None:
    session = _CaptureSession()

    snapshot_id = PaperWriteRepository(session).reset_demo_account()

    assert snapshot_id == "demo-reset"
    snapshot_params = next(
        item for item in session.params if item.get("snapshot_id") == snapshot_id
    )
    assert snapshot_params["event_time"].tzinfo is UTC
