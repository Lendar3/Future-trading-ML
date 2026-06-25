from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.backtest.costs import CostAssumptions
from app.backtest.engine import run_next_bar_backtest


@dataclass(frozen=True)
class WalkForwardFoldResult:
    fold_index: int
    train_start_time: pd.Timestamp
    train_end_time: pd.Timestamp
    test_start_time: pd.Timestamp
    test_end_time: pd.Timestamp
    train_bars: int
    test_bars: int
    signal_rows: int
    trade_rows: int
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "train_start_time": self.train_start_time.isoformat(),
            "train_end_time": self.train_end_time.isoformat(),
            "test_start_time": self.test_start_time.isoformat(),
            "test_end_time": self.test_end_time.isoformat(),
            "train_bars": self.train_bars,
            "test_bars": self.test_bars,
            "signal_rows": self.signal_rows,
            "trade_rows": self.trade_rows,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class WalkForwardBacktestResult:
    folds: list[WalkForwardFoldResult]
    aggregate_metrics: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "folds": [fold.to_dict() for fold in self.folds],
            "aggregate_metrics": self.aggregate_metrics,
        }


def _resolve_test_bars(
    total_bars: int, n_folds: int, min_train_bars: int, test_bars: int | None
) -> int:
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1.")
    if min_train_bars < 1:
        raise ValueError("min_train_bars must be >= 1.")
    if total_bars <= min_train_bars:
        raise ValueError(
            "Not enough bars for walk-forward: "
            f"total_bars={total_bars}, min_train_bars={min_train_bars}."
        )

    if test_bars is None:
        remaining = total_bars - min_train_bars
        inferred = remaining // n_folds
        if inferred < 1:
            raise ValueError(
                "Not enough bars for walk-forward: computed test_bars < 1. "
                f"total_bars={total_bars}, min_train_bars={min_train_bars}, n_folds={n_folds}."
            )
        test_bars = inferred
    elif test_bars < 1:
        raise ValueError("test_bars must be >= 1 when provided.")

    required = min_train_bars + (n_folds * test_bars)
    if total_bars < required:
        raise ValueError(
            "Not enough bars for walk-forward: "
            f"need at least {required}, got {total_bars} "
            f"(n_folds={n_folds}, min_train_bars={min_train_bars}, test_bars={test_bars})."
        )

    return test_bars


def run_walk_forward_backtest(
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
    n_folds: int = 3,
    min_train_bars: int = 100,
    test_bars: int | None = None,
) -> WalkForwardBacktestResult:
    bars_required = {"event_time", "open", "high", "low", "close"}
    bars_missing = bars_required.difference(bars.columns)
    if bars_missing:
        raise ValueError(f"Bars missing required columns: {sorted(bars_missing)}")
    if not signals.empty:
        signal_required = {"event_time", "side", "entry_ref", "take_ref", "stop_ref", "symbol"}
        signal_missing = signal_required.difference(signals.columns)
        if signal_missing:
            raise ValueError(f"Signals missing required columns: {sorted(signal_missing)}")

    bars_sorted = bars.sort_values("event_time").reset_index(drop=True)
    signals_sorted = signals.sort_values("event_time").reset_index(drop=True)
    resolved_test_bars = _resolve_test_bars(len(bars_sorted), n_folds, min_train_bars, test_bars)

    fold_results: list[WalkForwardFoldResult] = []
    for fold_index in range(n_folds):
        train_end_index = min_train_bars + (fold_index * resolved_test_bars)
        test_start_index = train_end_index
        test_end_index = test_start_index + resolved_test_bars

        train_window = bars_sorted.iloc[:train_end_index]
        test_window = bars_sorted.iloc[test_start_index:test_end_index]
        train_start_time = pd.Timestamp(train_window.iloc[0]["event_time"])
        train_end_time = pd.Timestamp(train_window.iloc[-1]["event_time"])
        test_start_time = pd.Timestamp(test_window.iloc[0]["event_time"])
        test_end_time = pd.Timestamp(test_window.iloc[-1]["event_time"])

        fold_signals = signals_sorted[
            (signals_sorted["event_time"] >= test_start_time)
            & (signals_sorted["event_time"] <= test_end_time)
        ]

        fold_backtest = run_next_bar_backtest(
            signals=fold_signals,
            bars=test_window,
            costs=costs,
            initial_equity=initial_equity,
            position_fraction=position_fraction,
            max_holding_bars=max_holding_bars,
            max_concurrent_positions=max_concurrent_positions,
            max_positions_per_symbol=max_positions_per_symbol,
            cooldown_bars_after_stop=cooldown_bars_after_stop,
            allow_opposite_side_same_symbol=allow_opposite_side_same_symbol,
            funding_rate=funding_rate,
            apply_funding_schedule=apply_funding_schedule,
        )

        fold_results.append(
            WalkForwardFoldResult(
                fold_index=fold_index,
                train_start_time=train_start_time,
                train_end_time=train_end_time,
                test_start_time=test_start_time,
                test_end_time=test_end_time,
                train_bars=len(train_window),
                test_bars=len(test_window),
                signal_rows=len(fold_signals),
                trade_rows=int(fold_backtest.metrics["trade_count"]),
                metrics=fold_backtest.metrics,
            )
        )

    total_returns = [fold.metrics.get("total_return", 0.0) for fold in fold_results]
    hit_rates = [fold.metrics.get("hit_rate", 0.0) for fold in fold_results]
    profit_factors = [fold.metrics.get("profit_factor", 0.0) for fold in fold_results]
    drawdowns = [fold.metrics.get("max_drawdown", 0.0) for fold in fold_results]
    trade_counts = [fold.trade_rows for fold in fold_results]

    aggregate_metrics = {
        "wf_fold_count": float(len(fold_results)),
        "wf_total_trades": float(sum(trade_counts)),
        "wf_avg_total_return": float(sum(total_returns) / len(fold_results)),
        "wf_avg_hit_rate": float(sum(hit_rates) / len(fold_results)),
        "wf_avg_profit_factor": float(sum(profit_factors) / len(fold_results)),
        "wf_worst_drawdown": float(min(drawdowns)),
        "wf_positive_fold_ratio": float(
            sum(1 for value in total_returns if value > 0) / len(fold_results)
        ),
    }

    return WalkForwardBacktestResult(
        folds=fold_results,
        aggregate_metrics=aggregate_metrics,
    )
