from __future__ import annotations

from app.paper.positions import apply_paper_close_to_positions, apply_paper_fill_to_positions


def test_apply_fill_opens_long_position() -> None:
    positions = apply_paper_fill_to_positions(
        current_positions={},
        fill_payload={
            "fill_id": "fill-long-001",
            "symbol": "BTCUSDT",
            "side": "buy_open_long",
            "quantity": 0.2,
            "price": 100.0,
            "filled_at": "2026-01-01T00:00:00+00:00",
        },
    )

    assert positions["BTCUSDT"]["side"] == "long"
    assert positions["BTCUSDT"]["status"] == "open"
    assert positions["BTCUSDT"]["quantity"] == 0.2


def test_apply_fill_opens_short_position() -> None:
    positions = apply_paper_fill_to_positions(
        current_positions={},
        fill_payload={
            "fill_id": "fill-short-001",
            "symbol": "ETHUSDT",
            "side": "sell_open_short",
            "quantity": 1.0,
            "price": 50.0,
            "filled_at": "2026-01-01T00:00:00+00:00",
        },
    )

    assert positions["ETHUSDT"]["side"] == "short"
    assert positions["ETHUSDT"]["status"] == "open"


def test_apply_close_removes_matching_open_position() -> None:
    positions = {
        "BTCUSDT": {
            "symbol": "BTCUSDT",
            "side": "long",
            "quantity": 0.2,
            "entry_price": 100.0,
            "notional": 20.0,
            "opened_at": "2026-01-01T00:00:00+00:00",
            "entry_fill_id": "fill-long-001",
            "status": "open",
        }
    }
    updated = apply_paper_close_to_positions(
        current_positions=positions,
        close_payload={
            "entry_fill_id": "fill-long-001",
            "symbol": "BTCUSDT",
            "side": "buy_open_long",
        },
    )

    assert "BTCUSDT" not in updated


def test_apply_fill_keeps_existing_open_position_on_opposite_side() -> None:
    positions = {
        "BTCUSDT": {
            "symbol": "BTCUSDT",
            "side": "long",
            "quantity": 0.2,
            "entry_price": 100.0,
            "notional": 20.0,
            "opened_at": "2026-01-01T00:00:00+00:00",
            "entry_fill_id": "fill-long-001",
            "status": "open",
        }
    }
    updated = apply_paper_fill_to_positions(
        current_positions=positions,
        fill_payload={
            "fill_id": "fill-short-001",
            "symbol": "BTCUSDT",
            "side": "sell_open_short",
            "quantity": 0.1,
            "price": 101.0,
            "filled_at": "2026-01-01T00:05:00+00:00",
        },
    )

    assert updated["BTCUSDT"]["side"] == "long"
    assert updated["BTCUSDT"]["entry_fill_id"] == "fill-long-001"
