from __future__ import annotations

import hashlib

from app.reports.paper_portfolio_audit import build_paper_portfolio_audit_report


def _close_snapshot_id(close_id: str) -> str:
    digest = hashlib.sha256(close_id.encode()).hexdigest()[:24]
    return f"psnap_close_{digest}"


def test_portfolio_audit_empty_state() -> None:
    report = build_paper_portfolio_audit_report(
        paper_fills=[],
        paper_trade_closes=[],
        portfolio_snapshots=[],
        current_portfolio_snapshot=None,
        limit=500,
    )

    assert report["summary"]["total_fills"] == 0
    assert report["summary"]["total_closes"] == 0
    assert report["open_positions"] == []
    assert report["open_fill_count"] == 0
    assert report["closed_trade_count"] == 0
    assert report["orphan_fills"] == []
    assert report["orphan_closes"] == []


def test_open_fill_has_matching_open_position() -> None:
    report = build_paper_portfolio_audit_report(
        paper_fills=[
            {
                "fill_id": "f1",
                "symbol": "BTCUSDT",
                "side": "buy_open_long",
                "pnl_status": "open",
            }
        ],
        paper_trade_closes=[],
        portfolio_snapshots=[],
        current_portfolio_snapshot={
            "snapshot_id": "s1",
            "event_time": "2026-01-01T00:00:00+00:00",
            "equity": 10000.0,
            "margin_used": 0.0,
            "available_margin": 10000.0,
            "positions": {
                "BTCUSDT": {
                    "symbol": "BTCUSDT",
                    "side": "long",
                    "quantity": 0.1,
                    "entry_price": 100.0,
                    "notional": 10.0,
                    "opened_at": "2026-01-01T00:00:00+00:00",
                    "entry_fill_id": "f1",
                    "status": "open",
                }
            },
        },
        limit=500,
    )

    assert report["open_fill_count"] == 1
    assert len(report["open_positions"]) == 1
    assert report["orphan_fills"] == []


def test_closed_fill_has_no_remaining_open_position() -> None:
    close_id = "c1"
    report = build_paper_portfolio_audit_report(
        paper_fills=[
            {
                "fill_id": "f1",
                "symbol": "BTCUSDT",
                "side": "buy_open_long",
                "pnl_status": "closed",
            }
        ],
        paper_trade_closes=[
            {
                "close_id": close_id,
                "entry_fill_id": "f1",
                "symbol": "BTCUSDT",
                "realized_pnl": 5.0,
            }
        ],
        portfolio_snapshots=[
            {
                "snapshot_id": "prev",
                "event_time": "2026-01-01T00:00:00+00:00",
                "equity": 10000.0,
                "margin_used": 0.0,
                "available_margin": 10000.0,
                "positions": {
                    "BTCUSDT": {
                        "symbol": "BTCUSDT",
                        "entry_fill_id": "f1",
                        "status": "open",
                    }
                },
            },
            {
                "snapshot_id": _close_snapshot_id(close_id),
                "event_time": "2026-01-01T01:00:00+00:00",
                "equity": 10005.0,
                "margin_used": 0.0,
                "available_margin": 10005.0,
                "positions": {},
            },
        ],
        current_portfolio_snapshot={
            "snapshot_id": _close_snapshot_id(close_id),
            "event_time": "2026-01-01T01:00:00+00:00",
            "equity": 10005.0,
            "margin_used": 0.0,
            "available_margin": 10005.0,
            "positions": {},
        },
        limit=500,
    )

    assert report["open_positions"] == []
    assert report["closed_trade_count"] == 1
    assert report["open_fill_count"] == 0


def test_orphan_fill_warning() -> None:
    report = build_paper_portfolio_audit_report(
        paper_fills=[
            {
                "fill_id": "f_orphan",
                "symbol": "BTCUSDT",
                "side": "buy_open_long",
                "pnl_status": "open",
            }
        ],
        paper_trade_closes=[],
        portfolio_snapshots=[],
        current_portfolio_snapshot={
            "snapshot_id": "s1",
            "event_time": "2026-01-01T00:00:00+00:00",
            "equity": 10000.0,
            "margin_used": 0.0,
            "available_margin": 10000.0,
            "positions": {},
        },
        limit=500,
    )

    assert len(report["orphan_fills"]) == 1
    assert any(
        "Open fills without matching positions detected." in warning
        for warning in report["position_consistency_warnings"]
    )


def test_orphan_close_warning() -> None:
    report = build_paper_portfolio_audit_report(
        paper_fills=[],
        paper_trade_closes=[
            {
                "close_id": "close_orphan",
                "entry_fill_id": "missing_fill",
                "symbol": "BTCUSDT",
                "realized_pnl": 1.0,
            }
        ],
        portfolio_snapshots=[],
        current_portfolio_snapshot={
            "snapshot_id": "s1",
            "event_time": "2026-01-01T00:00:00+00:00",
            "equity": 10000.0,
            "margin_used": 0.0,
            "available_margin": 10000.0,
            "positions": {},
        },
        limit=500,
    )

    assert len(report["orphan_closes"]) == 1
    assert any(
        "Closes referencing missing fills detected." in warning
        for warning in report["position_consistency_warnings"]
    )


def test_equity_consistency_after_close_matches_previous_equity_plus_realized_pnl() -> None:
    close_id = "close_eq_001"
    report = build_paper_portfolio_audit_report(
        paper_fills=[
            {
                "fill_id": "f1",
                "symbol": "BTCUSDT",
                "pnl_status": "closed",
            }
        ],
        paper_trade_closes=[
            {
                "close_id": close_id,
                "entry_fill_id": "f1",
                "symbol": "BTCUSDT",
                "realized_pnl": 12.0,
            }
        ],
        portfolio_snapshots=[
            {
                "snapshot_id": "prev",
                "event_time": "2026-01-01T00:00:00+00:00",
                "equity": 10000.0,
                "margin_used": 0.0,
                "available_margin": 10000.0,
                "positions": {},
            },
            {
                "snapshot_id": _close_snapshot_id(close_id),
                "event_time": "2026-01-01T01:00:00+00:00",
                "equity": 10012.0,
                "margin_used": 0.0,
                "available_margin": 10012.0,
                "positions": {},
            },
        ],
        current_portfolio_snapshot={
            "snapshot_id": _close_snapshot_id(close_id),
            "event_time": "2026-01-01T01:00:00+00:00",
            "equity": 10012.0,
            "margin_used": 0.0,
            "available_margin": 10012.0,
            "positions": {},
        },
        limit=500,
    )

    assert report["equity_consistency"]["all_consistent"] is True
    assert report["equity_consistency"]["checked_closes"] == 1
