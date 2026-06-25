from __future__ import annotations

from app.reports.paper_track_record import build_paper_track_record_report


def test_track_record_handles_empty_orders_and_fills() -> None:
    report = build_paper_track_record_report(
        paper_orders=[],
        paper_fills=[],
        portfolio_snapshots=[],
        current_portfolio_snapshot=None,
        limit=500,
    )

    assert report["summary"]["total_orders"] == 0
    assert report["summary"]["total_fills"] == 0
    assert report["summary"]["filled_trades"] == 0
    assert report["performance_metrics"]["total_pnl"] is None


def test_track_record_counts_fills_and_open_trades() -> None:
    report = build_paper_track_record_report(
        paper_orders=[
            {
                "order_id": "o1",
                "symbol": "BTCUSDT",
                "status": "filled",
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        ],
        paper_fills=[
            {
                "fill_id": "f1",
                "order_id": "o1",
                "symbol": "BTCUSDT",
                "side": "buy_open_long",
                "quantity": 1.0,
                "price": 100.0,
                "fee": 0.1,
                "filled_at": "2026-01-01T00:00:00+00:00",
                "notional": 100.0,
                "fees_paid": 0.1,
                "pnl_status": "open",
                "realized_pnl": None,
                "unrealized_pnl": None,
                "entry_price": 100.0,
                "entry_time": "2026-01-01T00:00:00+00:00",
            }
        ],
        portfolio_snapshots=[],
        current_portfolio_snapshot={"equity": 10000.0, "event_time": "2026-01-01T01:00:00+00:00"},
        limit=500,
    )

    summary = report["summary"]
    assert summary["total_orders"] == 1
    assert summary["total_fills"] == 1
    assert summary["filled_trades"] == 1
    assert summary["open_trades"] == 1
    assert summary["closed_trades"] == 0
    assert summary["pnl_status_counts"]["open"] == 1
    assert summary["total_notional"] == 100.0


def test_track_record_calculates_win_loss_metrics_for_closed_realized_pnl() -> None:
    report = build_paper_track_record_report(
        paper_orders=[],
        paper_fills=[
            {
                "fill_id": "f1",
                "order_id": "o1",
                "symbol": "BTCUSDT",
                "side": "buy_open_long",
                "quantity": 1.0,
                "price": 100.0,
                "fee": 0.1,
                "filled_at": "2026-01-01T00:00:00+00:00",
                "pnl_status": "closed",
                "realized_pnl": 10.0,
            },
            {
                "fill_id": "f2",
                "order_id": "o2",
                "symbol": "ETHUSDT",
                "side": "sell_open_short",
                "quantity": 1.0,
                "price": 50.0,
                "fee": 0.1,
                "filled_at": "2026-01-01T01:00:00+00:00",
                "pnl_status": "closed",
                "realized_pnl": -4.0,
            },
        ],
        portfolio_snapshots=[
            {
                "snapshot_id": "s1",
                "event_time": "2026-01-01T00:00:00+00:00",
                "equity": 10000.0,
                "margin_used": 0.0,
                "available_margin": 10000.0,
            },
            {
                "snapshot_id": "s2",
                "event_time": "2026-01-01T02:00:00+00:00",
                "equity": 10006.0,
                "margin_used": 0.0,
                "available_margin": 10006.0,
            },
        ],
        current_portfolio_snapshot={"equity": 10006.0, "event_time": "2026-01-01T02:00:00+00:00"},
        limit=500,
    )

    metrics = report["performance_metrics"]
    assert metrics["total_pnl"] == 6.0
    assert metrics["win_count"] == 1
    assert metrics["loss_count"] == 1
    assert metrics["win_rate"] == 0.5
    assert metrics["avg_win"] == 10.0
    assert metrics["avg_loss"] == -4.0
    assert metrics["avg_win_loss_ratio"] == 2.5
    assert metrics["profit_factor"] == 2.5


def test_track_record_win_loss_metrics_null_when_no_closed_trades() -> None:
    report = build_paper_track_record_report(
        paper_orders=[],
        paper_fills=[
            {
                "fill_id": "f1",
                "order_id": "o1",
                "symbol": "BTCUSDT",
                "side": "buy_open_long",
                "quantity": 1.0,
                "price": 100.0,
                "fee": 0.1,
                "filled_at": "2026-01-01T00:00:00+00:00",
                "pnl_status": "open",
                "realized_pnl": None,
            }
        ],
        portfolio_snapshots=[],
        current_portfolio_snapshot={"equity": 10000.0, "event_time": "2026-01-01T00:00:00+00:00"},
        limit=500,
    )

    assert report["performance_metrics"]["total_pnl"] is None
    assert report["performance_metrics"]["win_rate"] is None
    assert any(
        "No closed paper trades with realized PnL yet." in warning for warning in report["warnings"]
    )


def test_track_record_uses_realized_pnl_only_for_closed_trades() -> None:
    report = build_paper_track_record_report(
        paper_orders=[],
        paper_fills=[
            {
                "fill_id": "f1",
                "order_id": "o1",
                "symbol": "BTCUSDT",
                "side": "buy_open_long",
                "quantity": 1.0,
                "price": 100.0,
                "fee": 0.1,
                "filled_at": "2026-01-01T00:00:00+00:00",
                "pnl_status": "open",
                "realized_pnl": 999.0,
            },
            {
                "fill_id": "f2",
                "order_id": "o2",
                "symbol": "ETHUSDT",
                "side": "sell_open_short",
                "quantity": 1.0,
                "price": 50.0,
                "fee": 0.1,
                "filled_at": "2026-01-01T01:00:00+00:00",
                "pnl_status": "closed",
                "realized_pnl": 3.0,
            },
        ],
        portfolio_snapshots=[],
        current_portfolio_snapshot={"equity": 10003.0, "event_time": "2026-01-01T01:00:00+00:00"},
        limit=500,
    )

    assert report["performance_metrics"]["total_pnl"] == 3.0


def test_track_record_builds_daily_summary_when_fill_timestamps_and_pnl_available() -> None:
    report = build_paper_track_record_report(
        paper_orders=[],
        paper_fills=[
            {
                "fill_id": "f1",
                "order_id": "o1",
                "symbol": "BTCUSDT",
                "side": "buy_open_long",
                "quantity": 1.0,
                "price": 100.0,
                "fee": 0.1,
                "filled_at": "2026-01-01T00:00:00+00:00",
                "pnl_status": "closed",
                "realized_pnl": 3.0,
            },
            {
                "fill_id": "f2",
                "order_id": "o2",
                "symbol": "BTCUSDT",
                "side": "buy_open_long",
                "quantity": 1.0,
                "price": 101.0,
                "fee": 0.1,
                "filled_at": "2026-01-01T01:00:00+00:00",
                "pnl_status": "closed",
                "realized_pnl": -1.0,
            },
            {
                "fill_id": "f3",
                "order_id": "o3",
                "symbol": "ETHUSDT",
                "side": "sell_open_short",
                "quantity": 1.0,
                "price": 50.0,
                "fee": 0.1,
                "filled_at": "2026-01-02T01:00:00+00:00",
                "pnl_status": "closed",
                "realized_pnl": 2.5,
            },
        ],
        portfolio_snapshots=[],
        current_portfolio_snapshot={"equity": 10004.5, "event_time": "2026-01-02T01:00:00+00:00"},
        limit=500,
    )

    daily = report["daily_summary"]
    assert len(daily) == 2
    assert daily[0]["date"] == "2026-01-01"
    assert daily[0]["fill_count"] == 2
    assert daily[0]["pnl"] == 2.0
    assert daily[1]["date"] == "2026-01-02"
    assert daily[1]["fill_count"] == 1
    assert daily[1]["pnl"] == 2.5


def test_track_record_warns_if_closed_trades_exist_but_snapshot_update_is_missing() -> None:
    report = build_paper_track_record_report(
        paper_orders=[],
        paper_fills=[
            {
                "fill_id": "f1",
                "order_id": "o1",
                "symbol": "BTCUSDT",
                "side": "buy_open_long",
                "quantity": 1.0,
                "price": 100.0,
                "fee": 0.1,
                "filled_at": "2026-01-01T00:00:00+00:00",
                "pnl_status": "closed",
                "realized_pnl": 3.0,
            }
        ],
        portfolio_snapshots=[],
        current_portfolio_snapshot={"equity": 10000.0, "event_time": "2026-01-01T00:00:00+00:00"},
        limit=500,
    )

    assert any(
        "Portfolio equity is not automatically updated by close simulation yet." in warning
        for warning in report["warnings"]
    )


def test_closed_trade_realized_pnl_is_attributed_to_exit_time_not_fill_time() -> None:
    report = build_paper_track_record_report(
        paper_orders=[],
        paper_fills=[
            {
                "fill_id": "f1",
                "order_id": "o1",
                "symbol": "BTCUSDT",
                "side": "buy_open_long",
                "quantity": 1.0,
                "price": 100.0,
                "fee": 0.1,
                "filled_at": "2026-01-01T10:00:00+00:00",
                "exit_time": "2026-01-02T11:00:00+00:00",
                "pnl_status": "closed",
                "realized_pnl": 5.0,
            }
        ],
        portfolio_snapshots=[],
        current_portfolio_snapshot={"equity": 10005.0, "event_time": "2026-01-02T11:00:00+00:00"},
        limit=500,
    )

    daily = {row["date"]: row for row in report["daily_summary"]}
    assert daily["2026-01-01"]["fill_count"] == 1
    assert daily["2026-01-01"]["pnl"] == 0.0
    assert daily["2026-01-02"]["fill_count"] == 0
    assert daily["2026-01-02"]["pnl"] == 5.0

    monthly = {row["month"]: row for row in report["monthly_summary"]}
    assert monthly["2026-01"]["pnl"] == 5.0


def test_closed_trade_uses_closed_at_when_exit_time_missing() -> None:
    report = build_paper_track_record_report(
        paper_orders=[],
        paper_fills=[
            {
                "fill_id": "f1",
                "order_id": "o1",
                "symbol": "ETHUSDT",
                "side": "sell_open_short",
                "quantity": 1.0,
                "price": 50.0,
                "fee": 0.1,
                "filled_at": "2026-01-01T10:00:00+00:00",
                "closed_at": "2026-02-01T12:00:00+00:00",
                "pnl_status": "closed",
                "realized_pnl": -2.0,
            }
        ],
        portfolio_snapshots=[],
        current_portfolio_snapshot={"equity": 9998.0, "event_time": "2026-02-01T12:00:00+00:00"},
        limit=500,
    )

    monthly = {row["month"]: row for row in report["monthly_summary"]}
    assert monthly["2026-01"]["fill_count"] == 1
    assert monthly["2026-01"]["pnl"] == 0.0
    assert monthly["2026-02"]["fill_count"] == 0
    assert monthly["2026-02"]["pnl"] == -2.0


def test_closed_trade_fallback_to_fill_time_emits_warning() -> None:
    report = build_paper_track_record_report(
        paper_orders=[],
        paper_fills=[
            {
                "fill_id": "f1",
                "order_id": "o1",
                "symbol": "SOLUSDT",
                "side": "buy_open_long",
                "quantity": 1.0,
                "price": 20.0,
                "fee": 0.1,
                "filled_at": "2026-03-03T10:00:00+00:00",
                "pnl_status": "closed",
                "realized_pnl": 1.5,
            }
        ],
        portfolio_snapshots=[],
        current_portfolio_snapshot={"equity": 10001.5, "event_time": "2026-03-03T10:00:00+00:00"},
        limit=500,
    )

    daily = {row["date"]: row for row in report["daily_summary"]}
    assert daily["2026-03-03"]["pnl"] == 1.5
    assert any(
        "Some closed trades are missing exit_time/closed_at; fallback attribution was used."
        in warning
        for warning in report["warnings"]
    )
