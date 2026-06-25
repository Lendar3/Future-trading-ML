from __future__ import annotations

import pytest

from app.paper.accounting import build_paper_close_accounting, build_paper_fill_accounting


def test_accounting_marks_entry_only_fill_as_open() -> None:
    accounting = build_paper_fill_accounting(
        fill_payload={
            "side": "buy_open_long",
            "quantity": 1.5,
            "price": 100.0,
            "fee": 0.2,
            "filled_at": "2026-01-01T00:00:00+00:00",
        }
    )

    assert accounting["pnl_status"] == "open"
    assert accounting["realized_pnl"] is None
    assert accounting["unrealized_pnl"] is None


def test_accounting_computes_notional_and_fees_paid() -> None:
    accounting = build_paper_fill_accounting(
        fill_payload={
            "side": "sell_open_short",
            "quantity": 2.0,
            "price": 50.0,
            "fee": 0.15,
            "filled_at": "2026-01-01T00:00:00+00:00",
        }
    )

    assert accounting["notional"] == 100.0
    assert accounting["fees_paid"] == 0.15
    assert accounting["entry_price"] == 50.0
    assert accounting["entry_time"] == "2026-01-01T00:00:00+00:00"


def test_close_accounting_long_realized_pnl() -> None:
    close = build_paper_close_accounting(
        entry_fill_payload={
            "fill_id": "f_long",
            "side": "buy_open_long",
            "quantity": 2.0,
            "price": 100.0,
            "fee": 0.1,
            "filled_at": "2026-01-01T00:00:00+00:00",
        },
        exit_price=110.0,
        exit_time="2026-01-01T01:00:00+00:00",
        exit_fee=0.2,
    )

    assert close["pnl_status"] == "closed"
    assert close["gross_pnl"] == 20.0
    assert close["total_fees_paid"] == pytest.approx(0.3)
    assert close["realized_pnl"] == pytest.approx(19.7)


def test_close_accounting_short_realized_pnl() -> None:
    close = build_paper_close_accounting(
        entry_fill_payload={
            "fill_id": "f_short",
            "side": "sell_open_short",
            "quantity": 1.5,
            "price": 120.0,
            "fee": 0.09,
            "filled_at": "2026-01-01T00:00:00+00:00",
        },
        exit_price=100.0,
        exit_time="2026-01-01T02:00:00+00:00",
        exit_fee=0.06,
    )

    assert close["gross_pnl"] == pytest.approx(30.0)
    assert close["total_fees_paid"] == pytest.approx(0.15)
    assert close["realized_pnl"] == pytest.approx(29.85)


def test_close_accounting_rejects_invalid_exit_price() -> None:
    with pytest.raises(ValueError, match="exit_price must be positive"):
        build_paper_close_accounting(
            entry_fill_payload={
                "fill_id": "f_invalid",
                "side": "buy_open_long",
                "quantity": 1.0,
                "price": 100.0,
                "fee": 0.1,
                "filled_at": "2026-01-01T00:00:00+00:00",
            },
            exit_price=0.0,
            exit_time="2026-01-01T00:10:00+00:00",
        )


def test_close_accounting_rejects_exit_time_before_entry_time() -> None:
    with pytest.raises(ValueError, match="exit_time must be greater than or equal to entry_time"):
        build_paper_close_accounting(
            entry_fill_payload={
                "fill_id": "f_time",
                "side": "sell_open_short",
                "quantity": 1.0,
                "price": 100.0,
                "fee": 0.1,
                "filled_at": "2026-01-01T00:00:00+00:00",
            },
            exit_price=99.0,
            exit_time="2025-12-31T23:59:59+00:00",
        )
