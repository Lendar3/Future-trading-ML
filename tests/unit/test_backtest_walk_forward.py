from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from app.backtest.walk_forward import run_walk_forward_backtest
from app.core.safety import LiveTradingDisabledError
from app.execution.real_disabled import RealBrokerInterface


def _bars(count: int) -> pd.DataFrame:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    price = 100.0
    for idx in range(count):
        event_time = base + timedelta(minutes=idx)
        close_price = price + (0.2 if idx % 2 == 0 else -0.1)
        rows.append(
            {
                "event_time": event_time,
                "open": price,
                "high": max(price, close_price) + 0.5,
                "low": min(price, close_price) - 0.5,
                "close": close_price,
            }
        )
        price = close_price
    return pd.DataFrame(rows)


def _signals_from_bars(bars: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in bars.itertuples(index=False):
        records.append(
            {
                "event_time": row.event_time,
                "side": "long",
                "entry_ref": float(row.open),
                "take_ref": float(row.open) * 1.02,
                "stop_ref": float(row.open) * 0.98,
                "symbol": "BTCUSDT",
                "ttl_sec": 120,
            }
        )
    return pd.DataFrame(records)


def test_walk_forward_creates_expected_folds_in_chronological_order() -> None:
    bars = _bars(180)
    signals = _signals_from_bars(bars)
    result = run_walk_forward_backtest(
        signals=signals,
        bars=bars,
        n_folds=3,
        min_train_bars=90,
        test_bars=20,
        max_holding_bars=2,
    )

    assert len(result.folds) == 3
    assert result.aggregate_metrics["wf_fold_count"] == 3.0
    assert result.folds[0].test_end_time < result.folds[1].test_start_time
    assert result.folds[1].test_end_time < result.folds[2].test_start_time
    assert result.folds[0].train_end_time < result.folds[0].test_start_time


def test_walk_forward_raises_on_not_enough_bars() -> None:
    bars = _bars(20)
    signals = _signals_from_bars(bars)
    with pytest.raises(ValueError, match="Not enough bars for walk-forward"):
        run_walk_forward_backtest(
            signals=signals,
            bars=bars,
            n_folds=3,
            min_train_bars=15,
            test_bars=3,
        )


def test_walk_forward_aggregate_metrics_are_present() -> None:
    bars = _bars(160)
    signals = _signals_from_bars(bars)
    result = run_walk_forward_backtest(
        signals=signals,
        bars=bars,
        n_folds=3,
        min_train_bars=100,
        test_bars=15,
    )
    metrics = result.aggregate_metrics
    assert "wf_total_trades" in metrics
    assert "wf_avg_total_return" in metrics
    assert "wf_avg_hit_rate" in metrics
    assert "wf_avg_profit_factor" in metrics
    assert "wf_worst_drawdown" in metrics
    assert "wf_positive_fold_ratio" in metrics


def test_walk_forward_with_funding_schedule_runs() -> None:
    bars = _bars(170)
    signals = _signals_from_bars(bars)
    result = run_walk_forward_backtest(
        signals=signals,
        bars=bars,
        n_folds=3,
        min_train_bars=100,
        test_bars=15,
        apply_funding_schedule=True,
        funding_rate=0.0005,
    )
    assert len(result.folds) == 3
    assert "funding_paid_bp_total" in result.folds[0].metrics
    assert "funding_events_total" in result.folds[0].metrics


def test_live_execution_remains_disabled() -> None:
    broker = RealBrokerInterface()
    with pytest.raises(LiveTradingDisabledError):
        broker.place_order({"symbol": "BTCUSDT", "side": "Buy", "qty": 0.001})
