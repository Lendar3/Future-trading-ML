from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, cast

import numpy as np
import pandas as pd

from app.backtest.costs import CostAssumptions, bp_to_return, funding_for_holding_window


@dataclass(frozen=True)
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: dict[str, float]


def _validate_backtest_inputs(
    *,
    position_fraction: float,
    max_holding_bars: int,
    max_concurrent_positions: int,
    max_positions_per_symbol: int,
    cooldown_bars_after_stop: int,
) -> None:
    if not 0.0 < position_fraction <= 1.0:
        raise ValueError("position_fraction must be in (0, 1].")
    if max_holding_bars < 1:
        raise ValueError("max_holding_bars must be >= 1.")
    if max_concurrent_positions < 1:
        raise ValueError("max_concurrent_positions must be >= 1.")
    if max_positions_per_symbol < 1:
        raise ValueError("max_positions_per_symbol must be >= 1.")
    if cooldown_bars_after_stop < 0:
        raise ValueError("cooldown_bars_after_stop must be >= 0.")


def _validate_required_columns(signals: pd.DataFrame, bars: pd.DataFrame) -> None:
    required = {"event_time", "side", "entry_ref", "take_ref", "stop_ref", "symbol"}
    missing = required.difference(signals.columns)
    if missing:
        raise ValueError(f"Signals missing required columns: {sorted(missing)}")
    bars_required = {"event_time", "open", "high", "low", "close"}
    bars_missing = bars_required.difference(bars.columns)
    if bars_missing:
        raise ValueError(f"Bars missing required columns: {sorted(bars_missing)}")


def _empty_backtest_result(initial_equity: float) -> BacktestResult:
    equity = pd.DataFrame({"equity": [initial_equity]})
    return BacktestResult(
        pd.DataFrame(),
        equity,
        {
            "total_return": 0.0,
            "net_return": 0.0,
            "trade_count": 0.0,
            "hit_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "final_equity": initial_equity,
            "fees_paid_bp_total": 0.0,
            "slippage_paid_bp_total": 0.0,
            "funding_paid_bp_total": 0.0,
            "funding_events_total": 0.0,
            "funding_return_total": 0.0,
            "total_cost_bp": 0.0,
            "signals_seen": 0.0,
            "signals_traded": 0.0,
            "signals_skipped_no_trade": 0.0,
            "signals_skipped_exposure": 0.0,
            "signals_skipped_cooldown": 0.0,
            "signals_skipped_no_data": 0.0,
            # Backward-compatible aliases.
            "fees_paid": 0.0,
            "slippage_cost": 0.0,
            "funding_paid": 0.0,
        },
    )


def _infer_bar_seconds(bars_sorted: pd.DataFrame) -> int:
    bar_seconds = 60
    if len(bars_sorted) >= 2:
        deltas = (
            pd.to_datetime(bars_sorted["event_time"]).diff().dropna().dt.total_seconds().to_numpy()
        )
        if len(deltas):
            bar_seconds = int(max(float(np.median(deltas)), 1.0))
    return bar_seconds


def _as_timestamp(value: object) -> pd.Timestamp:
    return pd.Timestamp(cast(Any, value))


def _resolve_ttl_bars(
    *,
    signal: Any,
    bar_seconds: int,
    max_holding_bars: int,
) -> int:
    ttl_sec = int(getattr(signal, "ttl_sec", 0) or 0)
    ttl_bars = max_holding_bars
    if ttl_sec > 0:
        ttl_bars = max(1, int(np.ceil(ttl_sec / bar_seconds)))
    return min(ttl_bars, max_holding_bars)


def _resolve_stop_take(
    *,
    signal: Any,
    entry_price: float,
    direction: int,
) -> tuple[float, float]:
    stop_ref = (
        float(signal.stop_ref)
        if float(signal.stop_ref) > 0
        else (entry_price * (1 - 0.01) if direction == 1 else entry_price * (1 + 0.01))
    )
    take_ref = (
        float(signal.take_ref)
        if float(signal.take_ref) > 0
        else (entry_price * (1 + 0.01) if direction == 1 else entry_price * (1 - 0.01))
    )
    return stop_ref, take_ref


def _simulate_trade_exit(
    *,
    direction: int,
    holding_window: pd.DataFrame,
    stop_ref: float,
    take_ref: float,
    entry_time: object,
    entry_price: float,
) -> tuple[float, object, str]:
    exit_price = entry_price
    exit_time = entry_time
    exit_reason = "no_exit_data"
    valid_bar_seen = False

    for bar in holding_window.itertuples(index=False):
        bar_any = cast(Any, bar)
        high = float(bar_any.high)
        low = float(bar_any.low)
        close = float(bar_any.close)
        if np.isnan(high) or np.isnan(low) or np.isnan(close):
            continue
        valid_bar_seen = True

        if direction == 1:
            stop_hit = low <= stop_ref
            take_hit = high >= take_ref
            if stop_hit:
                exit_price = stop_ref
                exit_time = bar_any.event_time
                exit_reason = "stop_loss"
                break
            if take_hit:
                exit_price = take_ref
                exit_time = bar_any.event_time
                exit_reason = "take_profit"
                break
        else:
            stop_hit = high >= stop_ref
            take_hit = low <= take_ref
            if stop_hit:
                exit_price = stop_ref
                exit_time = bar_any.event_time
                exit_reason = "stop_loss"
                break
            if take_hit:
                exit_price = take_ref
                exit_time = bar_any.event_time
                exit_reason = "take_profit"
                break

        exit_price = close
        exit_time = bar_any.event_time
        exit_reason = "ttl_exit"

    if not valid_bar_seen:
        exit_price = entry_price
        exit_time = entry_time
        exit_reason = "no_exit_data"

    return exit_price, exit_time, exit_reason


def _calculate_funding_and_costs(
    *,
    cost_model: CostAssumptions,
    apply_funding_schedule: bool,
    funding_rate: float | None,
    signal_side: str,
    entry_time: object,
    exit_time: object,
    gross_return: float,
) -> dict[str, float | int]:
    fees_bp = 2 * cost_model.taker_fee_bp
    slippage_bp = cost_model.spread_bp + cost_model.slippage_bp
    if apply_funding_schedule:
        resolved_funding_rate = float(funding_rate) if funding_rate is not None else 0.0
        funding_events, funding_return, funding_bp = funding_for_holding_window(
            entry_time=_as_timestamp(entry_time),
            exit_time=_as_timestamp(exit_time),
            side=signal_side,
            funding_rate=resolved_funding_rate,
        )
    else:
        funding_events = 0
        funding_bp = cost_model.funding_bp
        funding_return = -bp_to_return(funding_bp)
    trade_cost_bp = fees_bp + slippage_bp + funding_bp
    net_return = gross_return - bp_to_return(fees_bp + slippage_bp) + funding_return
    return {
        "fees_bp": fees_bp,
        "slippage_bp": slippage_bp,
        "funding_events": funding_events,
        "funding_return": funding_return,
        "funding_bp": funding_bp,
        "trade_cost_bp": trade_cost_bp,
        "net_return": net_return,
    }


def _build_trade_row(
    *,
    signal: Any,
    signal_event_time: pd.Timestamp,
    entry_time: object,
    exit_time: object,
    entry_price: float,
    exit_price: float,
    exit_reason: str,
    gross_return: float,
    net_return: float,
    funding_events: int,
    funding_return: float,
    funding_bp: float,
    position_notional: float,
    pnl: float,
    equity_after: float,
) -> dict[str, float | str | pd.Timestamp]:
    return {
        "symbol": signal.symbol,
        "side": signal.side,
        "signal_event_time": signal_event_time,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "gross_return": gross_return,
        "net_return": net_return,
        "funding_events": funding_events,
        "funding_return": funding_return,
        "funding_bp": funding_bp,
        "position_notional": position_notional,
        "pnl": pnl,
        "equity_after": equity_after,
    }


def _build_metrics(
    *,
    trade_frame: pd.DataFrame,
    equity_curve: pd.DataFrame,
    initial_equity: float,
    final_equity: float,
    fees_paid_bp_total: float,
    slippage_paid_bp_total: float,
    funding_paid_bp_total: float,
    funding_events_total: int,
    funding_return_total: float,
    total_cost_bp: float,
    signals_seen: int,
    signals_traded: int,
    signals_skipped_no_trade: int,
    signals_skipped_exposure: int,
    signals_skipped_cooldown: int,
    signals_skipped_no_data: int,
) -> dict[str, float]:
    total_return = (final_equity / initial_equity) - 1.0
    wins = int((trade_frame["pnl"] > 0).sum()) if not trade_frame.empty else 0
    losses = int((trade_frame["pnl"] < 0).sum()) if not trade_frame.empty else 0
    avg_win = float(trade_frame.loc[trade_frame["pnl"] > 0, "net_return"].mean()) if wins else 0.0
    avg_loss = (
        float(trade_frame.loc[trade_frame["pnl"] < 0, "net_return"].mean()) if losses else 0.0
    )
    gross_profit = (
        float(trade_frame.loc[trade_frame["pnl"] > 0, "pnl"].sum())
        if not trade_frame.empty
        else 0.0
    )
    gross_loss = (
        float(-trade_frame.loc[trade_frame["pnl"] < 0, "pnl"].sum())
        if not trade_frame.empty
        else 0.0
    )
    profit_factor = (
        (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    )
    if not equity_curve.empty and "equity" in equity_curve.columns:
        running_peak = equity_curve["equity"].cummax()
        drawdown = (equity_curve["equity"] / running_peak) - 1.0
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    else:
        max_drawdown = 0.0

    return {
        "total_return": total_return,
        "net_return": total_return,
        "trade_count": float(len(trade_frame)),
        "hit_rate": wins / len(trade_frame) if len(trade_frame) else 0.0,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "final_equity": final_equity,
        "fees_paid_bp_total": fees_paid_bp_total,
        "slippage_paid_bp_total": slippage_paid_bp_total,
        "funding_paid_bp_total": funding_paid_bp_total,
        "funding_events_total": float(funding_events_total),
        "funding_return_total": funding_return_total,
        "total_cost_bp": total_cost_bp,
        "signals_seen": float(signals_seen),
        "signals_traded": float(signals_traded),
        "signals_skipped_no_trade": float(signals_skipped_no_trade),
        "signals_skipped_exposure": float(signals_skipped_exposure),
        "signals_skipped_cooldown": float(signals_skipped_cooldown),
        "signals_skipped_no_data": float(signals_skipped_no_data),
        # Backward-compatible aliases.
        "fees_paid": fees_paid_bp_total,
        "slippage_cost": slippage_paid_bp_total,
        "funding_paid": funding_paid_bp_total,
    }


def run_next_bar_backtest(
    signals: pd.DataFrame,
    bars: pd.DataFrame,
    costs: CostAssumptions | None = None,
    initial_equity: float = 10_000.0,
    position_fraction: float = 0.10,
    max_holding_bars: int = 3,
    max_concurrent_positions: int = 1,
    max_positions_per_symbol: int = 1,
    cooldown_bars_after_stop: int = 0,
    allow_opposite_side_same_symbol: bool = False,
    funding_rate: float | None = None,
    apply_funding_schedule: bool = False,
) -> BacktestResult:
    cost_model = costs or CostAssumptions()
    _validate_backtest_inputs(
        position_fraction=position_fraction,
        max_holding_bars=max_holding_bars,
        max_concurrent_positions=max_concurrent_positions,
        max_positions_per_symbol=max_positions_per_symbol,
        cooldown_bars_after_stop=cooldown_bars_after_stop,
    )
    if signals.empty:
        return _empty_backtest_result(initial_equity)

    _validate_required_columns(signals, bars)

    bars_sorted = bars.sort_values("event_time").reset_index(drop=True)
    bar_seconds = _infer_bar_seconds(bars_sorted)

    trades: list[dict[str, float | str | pd.Timestamp]] = []
    equity = initial_equity
    curve: list[dict[str, object]] = []

    fees_paid_bp_total = 0.0
    slippage_paid_bp_total = 0.0
    funding_paid_bp_total = 0.0
    funding_events_total = 0
    funding_return_total = 0.0
    total_cost_bp = 0.0
    signals_seen = 0
    signals_traded = 0
    signals_skipped_no_trade = 0
    signals_skipped_exposure = 0
    signals_skipped_cooldown = 0
    signals_skipped_no_data = 0

    active_positions: list[dict[str, object]] = []
    cooldown_until_by_symbol: dict[str, pd.Timestamp] = {}

    for signal in signals.sort_values("event_time").itertuples(index=False):
        signal_any = cast(Any, signal)
        signals_seen += 1
        signal_event_time = _as_timestamp(signal_any.event_time)
        signal_symbol = str(signal_any.symbol)
        signal_side = str(signal_any.side)

        active_positions = [
            p for p in active_positions if _as_timestamp(p["exit_time"]) > signal_event_time
        ]

        if signal_any.side == "no_trade":
            signals_skipped_no_trade += 1
            continue

        cooldown_until = cooldown_until_by_symbol.get(signal_symbol)
        if cooldown_until is not None and signal_event_time < cooldown_until:
            signals_skipped_cooldown += 1
            continue

        if len(active_positions) >= max_concurrent_positions:
            signals_skipped_exposure += 1
            continue

        same_symbol_positions = [p for p in active_positions if p["symbol"] == signal_symbol]
        if len(same_symbol_positions) >= max_positions_per_symbol:
            signals_skipped_exposure += 1
            continue

        if not allow_opposite_side_same_symbol and any(
            str(p["side"]) != signal_side for p in same_symbol_positions
        ):
            signals_skipped_exposure += 1
            continue

        next_bars = bars_sorted[bars_sorted["event_time"] > signal_event_time]
        if next_bars.empty:
            signals_skipped_no_data += 1
            continue

        entry_bar = next_bars.iloc[0]
        entry_time = entry_bar["event_time"]
        entry_price = float(entry_bar["open"])
        holding_bars = _resolve_ttl_bars(
            signal=signal,
            bar_seconds=bar_seconds,
            max_holding_bars=max_holding_bars,
        )
        holding_window = next_bars.iloc[:holding_bars].reset_index(drop=True)

        direction = 1 if signal_any.side == "long" else -1
        stop_ref, take_ref = _resolve_stop_take(
            signal=signal,
            entry_price=entry_price,
            direction=direction,
        )
        exit_price, exit_time, exit_reason = _simulate_trade_exit(
            direction=direction,
            holding_window=holding_window,
            stop_ref=stop_ref,
            take_ref=take_ref,
            entry_time=entry_time,
            entry_price=entry_price,
        )

        gross_return = direction * ((exit_price / entry_price) - 1.0)
        funding_and_costs = _calculate_funding_and_costs(
            cost_model=cost_model,
            apply_funding_schedule=apply_funding_schedule,
            funding_rate=funding_rate,
            signal_side=str(signal_any.side),
            entry_time=entry_time,
            exit_time=exit_time,
            gross_return=gross_return,
        )

        position_notional = equity * position_fraction
        net_return = float(funding_and_costs["net_return"])
        pnl = position_notional * net_return
        equity_after = equity + pnl

        fees_paid_bp_total += float(funding_and_costs["fees_bp"])
        slippage_paid_bp_total += float(funding_and_costs["slippage_bp"])
        funding_paid_bp_total += float(funding_and_costs["funding_bp"])
        funding_events_total += int(funding_and_costs["funding_events"])
        funding_return_total += float(funding_and_costs["funding_return"])
        total_cost_bp += float(funding_and_costs["trade_cost_bp"])

        trades.append(
            _build_trade_row(
                signal=signal_any,
                signal_event_time=signal_event_time,
                entry_time=entry_time,
                exit_time=exit_time,
                entry_price=entry_price,
                exit_price=exit_price,
                exit_reason=exit_reason,
                gross_return=gross_return,
                net_return=net_return,
                funding_events=int(funding_and_costs["funding_events"]),
                funding_return=float(funding_and_costs["funding_return"]),
                funding_bp=float(funding_and_costs["funding_bp"]),
                position_notional=position_notional,
                pnl=pnl,
                equity_after=equity_after,
            )
        )
        signals_traded += 1
        active_positions.append(
            {
                "symbol": signal_symbol,
                "side": signal_side,
                "exit_time": _as_timestamp(exit_time),
            }
        )
        if exit_reason == "stop_loss" and cooldown_bars_after_stop > 0:
            cooldown_until_by_symbol[signal_symbol] = _as_timestamp(exit_time) + timedelta(
                seconds=bar_seconds * cooldown_bars_after_stop
            )
        equity = equity_after
        curve.append({"event_time": exit_time, "equity": equity})

    trade_frame = pd.DataFrame(trades)
    equity_curve = pd.DataFrame(curve) if curve else pd.DataFrame({"equity": [initial_equity]})
    final_equity = float(equity)
    metrics = _build_metrics(
        trade_frame=trade_frame,
        equity_curve=equity_curve,
        initial_equity=initial_equity,
        final_equity=final_equity,
        fees_paid_bp_total=fees_paid_bp_total,
        slippage_paid_bp_total=slippage_paid_bp_total,
        funding_paid_bp_total=funding_paid_bp_total,
        funding_events_total=funding_events_total,
        funding_return_total=funding_return_total,
        total_cost_bp=total_cost_bp,
        signals_seen=signals_seen,
        signals_traded=signals_traded,
        signals_skipped_no_trade=signals_skipped_no_trade,
        signals_skipped_exposure=signals_skipped_exposure,
        signals_skipped_cooldown=signals_skipped_cooldown,
        signals_skipped_no_data=signals_skipped_no_data,
    )
    return BacktestResult(trade_frame, equity_curve, metrics)
