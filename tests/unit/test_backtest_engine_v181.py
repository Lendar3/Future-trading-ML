from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from app.backtest.costs import CostAssumptions
from app.backtest.engine import run_next_bar_backtest
from app.core.safety import LiveTradingDisabledError
from app.execution.real_disabled import RealBrokerInterface


def _bars(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    out = []
    for idx, (open_price, high_price, low_price, close_price) in enumerate(rows, start=1):
        ts = base + timedelta(minutes=idx)
        out.append(
            {
                "event_time": ts,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
            }
        )
    return pd.DataFrame(out)


def _bars_from_start(
    start: datetime, rows: list[tuple[float, float, float, float]]
) -> pd.DataFrame:
    out = []
    for idx, (open_price, high_price, low_price, close_price) in enumerate(rows):
        ts = start + timedelta(minutes=idx)
        out.append(
            {
                "event_time": ts,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
            }
        )
    return pd.DataFrame(out)


def _signal(
    *,
    side: str,
    stop_ref: float,
    take_ref: float,
    event_offset_sec: int = 0,
    ttl_sec: int = 1800,
) -> pd.DataFrame:
    base = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=event_offset_sec)
    return pd.DataFrame(
        [
            {
                "event_time": base,
                "side": side,
                "entry_ref": 100.0,
                "take_ref": take_ref,
                "stop_ref": stop_ref,
                "symbol": "BTCUSDT",
                "ttl_sec": ttl_sec,
            }
        ]
    )


def _signals(rows: list[dict[str, object]]) -> pd.DataFrame:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    out: list[dict[str, object]] = []
    for row in rows:
        out.append(
            {
                "event_time": base + timedelta(seconds=int(row.get("event_offset_sec", 0))),
                "side": str(row["side"]),
                "entry_ref": float(row.get("entry_ref", 100.0)),
                "take_ref": float(row.get("take_ref", 120.0)),
                "stop_ref": float(row.get("stop_ref", 80.0)),
                "symbol": str(row.get("symbol", "BTCUSDT")),
                "ttl_sec": int(row.get("ttl_sec", 1800)),
            }
        )
    return pd.DataFrame(out)


def test_long_take_profit_hit() -> None:
    bars = _bars([(100, 105, 99, 104), (104, 104, 103, 103)])
    signals = _signal(side="long", stop_ref=95, take_ref=104)
    result = run_next_bar_backtest(signals, bars, initial_equity=10_000.0)
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "take_profit"
    assert float(trade["exit_price"]) == pytest.approx(104.0)


def test_long_stop_loss_hit() -> None:
    bars = _bars([(100, 103, 94, 95), (95, 96, 94, 95)])
    signals = _signal(side="long", stop_ref=95, take_ref=104)
    result = run_next_bar_backtest(signals, bars)
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "stop_loss"
    assert float(trade["exit_price"]) == pytest.approx(95.0)


def test_short_take_profit_hit() -> None:
    bars = _bars([(100, 101, 95, 96), (96, 96, 95, 95)])
    signals = _signal(side="short", stop_ref=103, take_ref=96)
    result = run_next_bar_backtest(signals, bars)
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "take_profit"
    assert float(trade["exit_price"]) == pytest.approx(96.0)


def test_short_stop_loss_hit() -> None:
    bars = _bars([(100, 104, 99, 103), (103, 103, 101, 102)])
    signals = _signal(side="short", stop_ref=103, take_ref=96)
    result = run_next_bar_backtest(signals, bars)
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "stop_loss"
    assert float(trade["exit_price"]) == pytest.approx(103.0)


def test_same_candle_stop_take_uses_conservative_stop_first() -> None:
    bars = _bars([(100, 105, 94, 100), (100, 101, 99, 100)])
    signals = _signal(side="long", stop_ref=95, take_ref=104)
    result = run_next_bar_backtest(signals, bars)
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "stop_loss"
    assert float(trade["exit_price"]) == pytest.approx(95.0)


def test_ttl_exit() -> None:
    bars = _bars([(100, 101, 99, 100.5), (100.5, 101, 100, 100.6), (100.6, 101, 100, 100.7)])
    signals = _signal(side="long", stop_ref=90, take_ref=120, ttl_sec=120)
    result = run_next_bar_backtest(signals, bars, max_holding_bars=5)
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "ttl_exit"
    # 120 sec on 1m bars -> exit on second bar in holding window.
    assert pd.to_datetime(trade["exit_time"]) == pd.to_datetime(bars.iloc[1]["event_time"])


def test_position_sizing_uses_fraction_of_equity() -> None:
    bars = _bars([(100, 105, 99, 104)])
    signals = _signal(side="long", stop_ref=95, take_ref=104)
    r_small = run_next_bar_backtest(signals, bars, initial_equity=10_000.0, position_fraction=0.10)
    r_big = run_next_bar_backtest(signals, bars, initial_equity=10_000.0, position_fraction=0.50)
    pnl_small = float(r_small.trades.iloc[0]["pnl"])
    pnl_big = float(r_big.trades.iloc[0]["pnl"])
    assert pnl_big == pytest.approx(pnl_small * 5, rel=1e-6)


def test_empty_signals_returns_zero_trade_metrics() -> None:
    bars = _bars([(100, 101, 99, 100)])
    result = run_next_bar_backtest(pd.DataFrame(), bars)
    assert result.metrics["trade_count"] == 0.0
    assert result.metrics["final_equity"] == pytest.approx(10_000.0)


def test_missing_required_signal_columns_raises() -> None:
    bars = _bars([(100, 101, 99, 100)])
    signals = pd.DataFrame([{"event_time": datetime(2026, 1, 1, tzinfo=UTC), "side": "long"}])
    with pytest.raises(ValueError, match="Signals missing required columns"):
        run_next_bar_backtest(signals, bars)


def test_no_exit_data_reason_when_prices_missing() -> None:
    bars = _bars([(100, float("nan"), float("nan"), float("nan"))])
    signals = _signal(side="long", stop_ref=95, take_ref=104)
    result = run_next_bar_backtest(signals, bars)
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "no_exit_data"


def test_live_execution_remains_disabled() -> None:
    broker = RealBrokerInterface()
    with pytest.raises(LiveTradingDisabledError):
        broker.place_order({"symbol": "BTCUSDT", "side": "Buy", "qty": 0.001})


def test_overlapping_signals_skipped_with_max_concurrent_positions_one() -> None:
    bars = _bars(
        [(100, 101, 99, 100), (100, 101, 99, 100), (100, 101, 99, 100), (100, 101, 99, 100)]
    )
    signals = _signals(
        [
            {"event_offset_sec": 0, "side": "long", "symbol": "BTCUSDT"},
            {"event_offset_sec": 90, "side": "long", "symbol": "ETHUSDT"},
        ]
    )
    result = run_next_bar_backtest(
        signals,
        bars,
        max_holding_bars=3,
        max_concurrent_positions=1,
        max_positions_per_symbol=2,
    )
    assert result.metrics["signals_seen"] == 2.0
    assert result.metrics["signals_traded"] == 1.0
    assert result.metrics["signals_skipped_exposure"] == 1.0


def test_same_symbol_second_signal_skipped_with_per_symbol_limit() -> None:
    bars = _bars(
        [(100, 101, 99, 100), (100, 101, 99, 100), (100, 101, 99, 100), (100, 101, 99, 100)]
    )
    signals = _signals(
        [
            {"event_offset_sec": 0, "side": "long", "symbol": "BTCUSDT"},
            {"event_offset_sec": 90, "side": "long", "symbol": "BTCUSDT"},
        ]
    )
    result = run_next_bar_backtest(
        signals,
        bars,
        max_holding_bars=3,
        max_concurrent_positions=3,
        max_positions_per_symbol=1,
    )
    assert result.metrics["signals_traded"] == 1.0
    assert result.metrics["signals_skipped_exposure"] == 1.0


def test_opposite_side_same_symbol_skipped_by_default() -> None:
    bars = _bars(
        [(100, 101, 99, 100), (100, 101, 99, 100), (100, 101, 99, 100), (100, 101, 99, 100)]
    )
    signals = _signals(
        [
            {"event_offset_sec": 0, "side": "long", "symbol": "BTCUSDT"},
            {"event_offset_sec": 90, "side": "short", "symbol": "BTCUSDT"},
        ]
    )
    result = run_next_bar_backtest(
        signals,
        bars,
        max_holding_bars=3,
        max_concurrent_positions=3,
        max_positions_per_symbol=2,
        allow_opposite_side_same_symbol=False,
    )
    assert result.metrics["signals_traded"] == 1.0
    assert result.metrics["signals_skipped_exposure"] == 1.0


def test_opposite_side_same_symbol_allowed_when_flag_true() -> None:
    bars = _bars(
        [(100, 101, 99, 100), (100, 101, 99, 100), (100, 101, 99, 100), (100, 101, 99, 100)]
    )
    signals = _signals(
        [
            {"event_offset_sec": 0, "side": "long", "symbol": "BTCUSDT"},
            {"event_offset_sec": 90, "side": "short", "symbol": "BTCUSDT"},
        ]
    )
    result = run_next_bar_backtest(
        signals,
        bars,
        max_holding_bars=3,
        max_concurrent_positions=3,
        max_positions_per_symbol=2,
        allow_opposite_side_same_symbol=True,
    )
    assert result.metrics["signals_traded"] == 2.0
    assert result.metrics["signals_skipped_exposure"] == 0.0


def test_cooldown_after_stop_loss_skips_later_same_symbol_signal() -> None:
    bars = _bars([(100, 101, 94, 96), (96, 97, 95, 96), (96, 97, 95, 96), (96, 97, 95, 96)])
    signals = _signals(
        [
            {
                "event_offset_sec": 0,
                "side": "long",
                "symbol": "BTCUSDT",
                "stop_ref": 95,
                "take_ref": 120,
            },
            {
                "event_offset_sec": 120,
                "side": "long",
                "symbol": "BTCUSDT",
                "stop_ref": 90,
                "take_ref": 120,
            },
        ]
    )
    result = run_next_bar_backtest(
        signals,
        bars,
        max_holding_bars=3,
        cooldown_bars_after_stop=2,
    )
    assert result.metrics["signals_traded"] == 1.0
    assert result.metrics["signals_skipped_cooldown"] == 1.0


def test_skip_metrics_track_no_trade_and_no_data() -> None:
    bars = _bars([(100, 101, 99, 100)])
    base = datetime(2026, 1, 1, tzinfo=UTC)
    signals = pd.DataFrame(
        [
            {
                "event_time": base,
                "side": "no_trade",
                "entry_ref": 100.0,
                "take_ref": 120.0,
                "stop_ref": 80.0,
                "symbol": "BTCUSDT",
                "ttl_sec": 1800,
            },
            {
                "event_time": base + timedelta(minutes=5),
                "side": "long",
                "entry_ref": 100.0,
                "take_ref": 120.0,
                "stop_ref": 80.0,
                "symbol": "BTCUSDT",
                "ttl_sec": 1800,
            },
        ]
    )
    result = run_next_bar_backtest(signals, bars)
    assert result.metrics["signals_seen"] == 2.0
    assert result.metrics["signals_traded"] == 0.0
    assert result.metrics["signals_skipped_no_trade"] == 1.0
    assert result.metrics["signals_skipped_no_data"] == 1.0


def test_long_pays_positive_funding_when_window_crossed() -> None:
    start = datetime(2026, 1, 1, 7, 59, tzinfo=UTC)
    bars = _bars_from_start(
        start, [(100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100)]
    )
    signals = _signal(
        side="long",
        stop_ref=90,
        take_ref=120,
        event_offset_sec=(7 * 3600) + (58 * 60),
        ttl_sec=180,
    )
    result = run_next_bar_backtest(
        signals=signals,
        bars=bars,
        costs=CostAssumptions(
            maker_fee_bp=0.0,
            taker_fee_bp=0.0,
            spread_bp=0.0,
            slippage_bp=0.0,
            funding_bp=0.0,
        ),
        apply_funding_schedule=True,
        funding_rate=0.001,
    )
    trade = result.trades.iloc[0]
    assert int(trade["funding_events"]) == 1
    assert float(trade["funding_return"]) == pytest.approx(-0.001)
    assert float(trade["funding_bp"]) == pytest.approx(10.0)
    assert result.metrics["funding_events_total"] == 1.0
    assert result.metrics["funding_return_total"] == pytest.approx(-0.001)
    assert result.metrics["funding_paid_bp_total"] == pytest.approx(10.0)


def test_long_receives_negative_funding_when_window_crossed() -> None:
    start = datetime(2026, 1, 1, 7, 59, tzinfo=UTC)
    bars = _bars_from_start(
        start, [(100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100)]
    )
    signals = _signal(
        side="long",
        stop_ref=90,
        take_ref=120,
        event_offset_sec=(7 * 3600) + (58 * 60),
        ttl_sec=180,
    )
    result = run_next_bar_backtest(
        signals=signals,
        bars=bars,
        costs=CostAssumptions(
            maker_fee_bp=0.0,
            taker_fee_bp=0.0,
            spread_bp=0.0,
            slippage_bp=0.0,
            funding_bp=0.0,
        ),
        apply_funding_schedule=True,
        funding_rate=-0.001,
    )
    trade = result.trades.iloc[0]
    assert int(trade["funding_events"]) == 1
    assert float(trade["funding_return"]) == pytest.approx(0.001)
    assert float(trade["funding_bp"]) == pytest.approx(-10.0)


def test_short_receives_positive_funding_when_window_crossed() -> None:
    start = datetime(2026, 1, 1, 7, 59, tzinfo=UTC)
    bars = _bars_from_start(
        start, [(100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100)]
    )
    signals = _signal(
        side="short",
        stop_ref=120,
        take_ref=90,
        event_offset_sec=(7 * 3600) + (58 * 60),
        ttl_sec=180,
    )
    result = run_next_bar_backtest(
        signals=signals,
        bars=bars,
        costs=CostAssumptions(
            maker_fee_bp=0.0,
            taker_fee_bp=0.0,
            spread_bp=0.0,
            slippage_bp=0.0,
            funding_bp=0.0,
        ),
        apply_funding_schedule=True,
        funding_rate=0.001,
    )
    trade = result.trades.iloc[0]
    assert int(trade["funding_events"]) == 1
    assert float(trade["funding_return"]) == pytest.approx(0.001)
    assert float(trade["funding_bp"]) == pytest.approx(-10.0)


def test_short_pays_negative_funding_when_window_crossed() -> None:
    start = datetime(2026, 1, 1, 7, 59, tzinfo=UTC)
    bars = _bars_from_start(
        start, [(100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100)]
    )
    signals = _signal(
        side="short",
        stop_ref=120,
        take_ref=90,
        event_offset_sec=(7 * 3600) + (58 * 60),
        ttl_sec=180,
    )
    result = run_next_bar_backtest(
        signals=signals,
        bars=bars,
        costs=CostAssumptions(
            maker_fee_bp=0.0,
            taker_fee_bp=0.0,
            spread_bp=0.0,
            slippage_bp=0.0,
            funding_bp=0.0,
        ),
        apply_funding_schedule=True,
        funding_rate=-0.001,
    )
    trade = result.trades.iloc[0]
    assert int(trade["funding_events"]) == 1
    assert float(trade["funding_return"]) == pytest.approx(-0.001)
    assert float(trade["funding_bp"]) == pytest.approx(10.0)


def test_no_funding_when_no_funding_window_crossed() -> None:
    start = datetime(2026, 1, 1, 7, 40, tzinfo=UTC)
    bars = _bars_from_start(
        start, [(100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100)]
    )
    signals = _signal(
        side="long",
        stop_ref=90,
        take_ref=120,
        event_offset_sec=(7 * 3600) + (39 * 60),
        ttl_sec=180,
    )
    result = run_next_bar_backtest(
        signals=signals,
        bars=bars,
        costs=CostAssumptions(
            maker_fee_bp=0.0,
            taker_fee_bp=0.0,
            spread_bp=0.0,
            slippage_bp=0.0,
            funding_bp=0.0,
        ),
        apply_funding_schedule=True,
        funding_rate=0.001,
    )
    trade = result.trades.iloc[0]
    assert int(trade["funding_events"]) == 0
    assert float(trade["funding_return"]) == pytest.approx(0.0)
    assert float(trade["funding_bp"]) == pytest.approx(0.0)


def test_legacy_flat_funding_behavior_unchanged_when_schedule_disabled() -> None:
    bars = _bars([(100, 100, 100, 100)])
    signals = _signal(side="long", stop_ref=90, take_ref=120, ttl_sec=60)
    result = run_next_bar_backtest(
        signals=signals,
        bars=bars,
        costs=CostAssumptions(
            maker_fee_bp=0.0,
            taker_fee_bp=0.0,
            spread_bp=0.0,
            slippage_bp=0.0,
            funding_bp=3.0,
        ),
        apply_funding_schedule=False,
        funding_rate=0.001,
    )
    trade = result.trades.iloc[0]
    assert float(trade["net_return"]) == pytest.approx(-0.0003)
    assert float(trade["funding_bp"]) == pytest.approx(3.0)
    assert float(trade["funding_return"]) == pytest.approx(-0.0003)
    assert result.metrics["funding_events_total"] == 0.0
    assert result.metrics["funding_paid_bp_total"] == pytest.approx(3.0)
