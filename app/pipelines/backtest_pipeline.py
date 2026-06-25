from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, cast

import pandas as pd
from sqlalchemy.orm import sessionmaker

from app.backtest.costs import CostAssumptions
from app.backtest.engine import run_next_bar_backtest
from app.backtest.walk_forward import run_walk_forward_backtest
from app.core.clock import utc_now
from app.features.dataset_builder import (
    FEATURE_SET_LEGACY,
    FEATURE_SET_PUBLIC,
    build_dataset_from_bars,
    build_dataset_from_public_tables,
)
from app.models.predict import predict_advisor_probabilities
from app.storage.database import session_scope
from app.storage.repositories import (
    PostgresBacktestRepository,
    PostgresMarketDataRepository,
    PostgresPublicMarketRepository,
)
from app.strategies.advisor_ml import predictions_to_signal_frame
from app.strategies.mean_reversion import generate_mean_reversion_signals


@dataclass(frozen=True)
class BacktestPipelineResult:
    run_id: str
    symbol: str
    timeframe: str
    dataset_rows: int
    signal_rows: int
    trade_rows: int
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "dataset_rows": self.dataset_rows,
            "signal_rows": self.signal_rows,
            "trade_rows": self.trade_rows,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class WalkForwardPipelineResult:
    symbol: str
    timeframe: str
    dataset_rows: int
    signal_rows: int
    fold_rows: int
    aggregate_metrics: dict[str, float]
    folds: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "dataset_rows": self.dataset_rows,
            "signal_rows": self.signal_rows,
            "fold_rows": self.fold_rows,
            "aggregate_metrics": self.aggregate_metrics,
            "folds": self.folds,
        }


@dataclass(frozen=True)
class MlWalkForwardPipelineResult:
    model_run_id: str
    symbol: str
    timeframe: str
    feature_set_version: str
    signal_count: int
    traded_signal_count: int
    aggregate_metrics: dict[str, float]
    folds: list[dict[str, object]]
    config: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "model_run_id": self.model_run_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "feature_set_version": self.feature_set_version,
            "signal_count": self.signal_count,
            "traded_signal_count": self.traded_signal_count,
            "aggregate_metrics": self.aggregate_metrics,
            "folds": self.folds,
            "config": self.config,
        }


@dataclass(frozen=True)
class StrategyComparisonPipelineResult:
    model_run_id: str
    symbol: str
    timeframe: str
    feature_set_version: str
    compared_strategies: list[str]
    comparison_rows: list[dict[str, object]]
    summary: dict[str, object]
    quality_gates: dict[str, bool]
    quality_gate_passed: bool
    quality_gate_warnings: list[str]
    comparison_config: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "model_run_id": self.model_run_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "feature_set_version": self.feature_set_version,
            "compared_strategies": self.compared_strategies,
            "comparison_rows": self.comparison_rows,
            "summary": self.summary,
            "quality_gates": self.quality_gates,
            "quality_gate_passed": self.quality_gate_passed,
            "quality_gate_warnings": self.quality_gate_warnings,
            "comparison_config": self.comparison_config,
        }


def _trade_rows(run_id: str, trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, trade in enumerate(trades):
        trade_id = hashlib.sha256(
            f"{run_id}-{index}-{trade['entry_time']}-{trade['side']}".encode()
        ).hexdigest()[:24]
        rows.append(
            {
                "trade_id": trade_id,
                "run_id": run_id,
                "symbol": trade["symbol"],
                "side": trade["side"],
                "entry_time": trade["entry_time"],
                "exit_time": trade["exit_time"],
                "pnl": float(trade["pnl"]),
            }
        )
    return rows


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _build_strategy_row(
    *,
    strategy_name: str,
    signal_count: int,
    assumptions_signature: str,
    wf_result: Any,
    initial_equity: float,
) -> dict[str, object]:
    fold_metrics = [fold.metrics for fold in wf_result.folds]
    fold_returns = [_as_float(m.get("total_return", 0.0)) for m in fold_metrics]
    fold_final_equities = [_as_float(m.get("final_equity", initial_equity)) for m in fold_metrics]
    fold_profit_factors = [_as_float(m.get("profit_factor", 0.0)) for m in fold_metrics]
    fold_hit_rates = [_as_float(m.get("hit_rate", 0.0)) for m in fold_metrics]
    fold_drawdowns = [_as_float(m.get("max_drawdown", 0.0)) for m in fold_metrics]
    fold_trade_counts = [int(_as_float(m.get("trade_count", 0.0))) for m in fold_metrics]

    fold_count = len(fold_returns)
    sum_fold_total_return = float(sum(fold_returns))
    avg_fold_total_return = float(sum_fold_total_return / fold_count) if fold_count else 0.0
    median_fold_total_return = float(median(fold_returns)) if fold_count else 0.0
    worst_fold_total_return = float(min(fold_returns)) if fold_count else 0.0
    best_fold_total_return = float(max(fold_returns)) if fold_count else 0.0
    avg_fold_final_equity = (
        float(sum(fold_final_equities) / fold_count) if fold_count else float(initial_equity)
    )
    positive_fold_count = sum(1 for value in fold_returns if value > 0.0)
    negative_fold_count = sum(1 for value in fold_returns if value < 0.0)
    zero_trade_fold_count = sum(1 for value in fold_trade_counts if value == 0)

    return {
        "strategy_name": strategy_name,
        "signal_count": int(signal_count),
        "trade_count": int(sum(fold_trade_counts)),
        # Backward-compatible fields from v1.8.6.
        "total_return": sum_fold_total_return,
        "hit_rate": (float(sum(fold_hit_rates) / fold_count) if fold_count else 0.0),
        "profit_factor": (float(sum(fold_profit_factors) / fold_count) if fold_count else 0.0),
        "max_drawdown": (float(min(fold_drawdowns)) if fold_count else 0.0),
        "final_equity": avg_fold_final_equity,
        # v1.8.7 explicit fold aggregates.
        "sum_fold_total_return": sum_fold_total_return,
        "avg_fold_total_return": avg_fold_total_return,
        "median_fold_total_return": median_fold_total_return,
        "worst_fold_total_return": worst_fold_total_return,
        "best_fold_total_return": best_fold_total_return,
        "avg_fold_final_equity": avg_fold_final_equity,
        "fold_count": fold_count,
        "positive_fold_count": positive_fold_count,
        "negative_fold_count": negative_fold_count,
        "zero_trade_fold_count": zero_trade_fold_count,
        "wf_positive_fold_ratio": (float(positive_fold_count / fold_count) if fold_count else 0.0),
        "assumptions_signature": assumptions_signature,
        "legacy_total_return_definition": "sum_fold_total_return",
        "legacy_final_equity_definition": "avg_fold_final_equity",
    }


def _evaluate_quality_gates(
    *,
    comparison_rows: list[dict[str, object]],
    summary: dict[str, object],
    min_fold_count: int = 2,
    min_positive_fold_ratio: float = 0.5,
) -> tuple[dict[str, bool], bool, list[str]]:
    ml_row = next(
        (row for row in comparison_rows if row.get("strategy_name") == "ml_advisor"),
        None,
    )
    rule_rows = [row for row in comparison_rows if row.get("strategy_name") != "ml_advisor"]
    if ml_row is None or not rule_rows:
        gates = {
            "has_trades_all_strategies": False,
            "ml_has_trades": False,
            "rule_has_trades": False,
            "ml_beats_best_rule_total_return": False,
            "ml_beats_best_rule_profit_factor": False,
            "ml_drawdown_not_worse_than_rule": False,
            "enough_folds": False,
            "positive_fold_ratio_ok": False,
        }
        early_warnings = ["Comparison rows are incomplete for quality gate evaluation."]
        return gates, False, early_warnings

    best_rule_by_return = max(rule_rows, key=lambda row: _as_float(row.get("total_return", 0.0)))
    best_rule_by_pf = max(rule_rows, key=lambda row: _as_float(row.get("profit_factor", 0.0)))
    best_rule_by_drawdown = max(rule_rows, key=lambda row: _as_float(row.get("max_drawdown", 0.0)))

    ml_trade_count = int(_as_float(ml_row.get("trade_count", 0.0)))
    ml_fold_count = int(_as_float(ml_row.get("fold_count", 0.0)))
    ml_positive_fold_ratio = _as_float(ml_row.get("wf_positive_fold_ratio", 0.0))

    has_trades_all_strategies = all(
        int(_as_float(row.get("trade_count", 0.0))) > 0 for row in comparison_rows
    )
    ml_has_trades = ml_trade_count > 0
    rule_has_trades = any(int(_as_float(row.get("trade_count", 0.0))) > 0 for row in rule_rows)
    ml_beats_best_rule_total_return = (
        _as_float(summary.get("ml_vs_best_rule_total_return_delta", 0.0)) > 0.0
    )
    ml_beats_best_rule_profit_factor = (
        _as_float(summary.get("ml_vs_best_rule_profit_factor_delta", 0.0)) > 0.0
    )
    ml_drawdown_not_worse_than_rule = _as_float(ml_row.get("max_drawdown", 0.0)) >= _as_float(
        best_rule_by_drawdown.get("max_drawdown", 0.0)
    )
    enough_folds = ml_fold_count >= min_fold_count
    positive_fold_ratio_ok = ml_positive_fold_ratio >= min_positive_fold_ratio

    gates = {
        "has_trades_all_strategies": has_trades_all_strategies,
        "ml_has_trades": ml_has_trades,
        "rule_has_trades": rule_has_trades,
        "ml_beats_best_rule_total_return": ml_beats_best_rule_total_return,
        "ml_beats_best_rule_profit_factor": ml_beats_best_rule_profit_factor,
        "ml_drawdown_not_worse_than_rule": ml_drawdown_not_worse_than_rule,
        "enough_folds": enough_folds,
        "positive_fold_ratio_ok": positive_fold_ratio_ok,
    }

    warnings: list[str] = []
    if not ml_has_trades:
        warnings.append("ML advisor produced no trades in evaluated folds.")
    if not rule_has_trades:
        warnings.append("Rule baseline produced no trades in evaluated folds.")
    if not ml_beats_best_rule_total_return:
        warnings.append(
            "ML total return does not beat best rule baseline "
            f"({best_rule_by_return.get('strategy_name', '')})."
        )
    if not ml_beats_best_rule_profit_factor:
        warnings.append(
            "ML profit factor does not beat best rule baseline "
            f"({best_rule_by_pf.get('strategy_name', '')})."
        )
    if not ml_drawdown_not_worse_than_rule:
        warnings.append("ML drawdown is worse than the best rule baseline drawdown.")
    if not enough_folds:
        warnings.append(f"Not enough folds for robust comparison (need >= {min_fold_count}).")
    if not positive_fold_ratio_ok:
        warnings.append(f"ML positive fold ratio below threshold {min_positive_fold_ratio:.2f}.")

    quality_gate_passed = all(gates.values())
    return gates, quality_gate_passed, warnings


def _load_bars_and_dataset(
    *,
    session_factory: sessionmaker,
    symbol: str,
    timeframe: str,
    source: str | None,
    feature_set_version: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    with session_scope(session_factory) as session:
        market_repo = PostgresMarketDataRepository(session)
        bars = market_repo.fetch_raw_bars(symbol=symbol, timeframe=timeframe, source=source)
        if bars.empty:
            raise ValueError(f"No raw bars found for {symbol} {timeframe}.")

        if feature_set_version == FEATURE_SET_PUBLIC:
            public_repo = PostgresPublicMarketRepository(session)
            dataset = build_dataset_from_public_tables(
                bars=bars,
                symbol=symbol,
                timeframe=timeframe,
                mark_bars=public_repo.fetch_mark_bars(symbol=symbol, timeframe=timeframe),
                index_bars=public_repo.fetch_index_bars(symbol=symbol, timeframe=timeframe),
                funding=public_repo.fetch_funding(symbol=symbol, source=source),
                open_interest=public_repo.fetch_open_interest(symbol=symbol, source=source),
                long_short_ratio=public_repo.fetch_long_short_ratio(symbol=symbol, source=source),
                feature_set_name=feature_set_version,
            )
        elif feature_set_version == FEATURE_SET_LEGACY:
            dataset = build_dataset_from_bars(
                bars=bars,
                symbol=symbol,
                timeframe=timeframe,
                feature_set_name=feature_set_version,
                label_horizon=3,
            )
        else:
            raise ValueError(
                f"Unsupported feature_set_version={feature_set_version!r}. "
                f"Expected {FEATURE_SET_LEGACY!r} or {FEATURE_SET_PUBLIC!r}."
            )

    return bars, dataset


def run_backtest_pipeline_from_postgres(
    session_factory: sessionmaker,
    symbol: str,
    timeframe: str,
    source: str | None = None,
    initial_equity: float = 10_000.0,
    position_fraction: float = 0.10,
    max_holding_bars: int = 3,
    max_concurrent_positions: int = 1,
    max_positions_per_symbol: int = 1,
    cooldown_bars_after_stop: int = 0,
    allow_opposite_side_same_symbol: bool = False,
    funding_rate: float | None = None,
    apply_funding_schedule: bool = False,
) -> BacktestPipelineResult:
    with session_scope(session_factory) as session:
        market_repo = PostgresMarketDataRepository(session)
        bars = market_repo.fetch_raw_bars(symbol=symbol, timeframe=timeframe, source=source)

    if bars.empty:
        raise ValueError(f"No raw bars found for {symbol} {timeframe}.")

    dataset = build_dataset_from_bars(
        bars=bars,
        symbol=symbol,
        timeframe=timeframe,
        label_horizon=3,
    )
    signals = generate_mean_reversion_signals(dataset, symbol=symbol)
    result = run_next_bar_backtest(
        signals=signals,
        bars=bars,
        costs=CostAssumptions(),
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
    config = {
        "symbol": symbol,
        "timeframe": timeframe,
        "source": source,
        "strategy": "mean_reversion",
        "execution": "next_bar",
        "initial_equity": initial_equity,
        "position_fraction": position_fraction,
        "max_holding_bars": max_holding_bars,
        "max_concurrent_positions": max_concurrent_positions,
        "max_positions_per_symbol": max_positions_per_symbol,
        "cooldown_bars_after_stop": cooldown_bars_after_stop,
        "allow_opposite_side_same_symbol": allow_opposite_side_same_symbol,
        "funding_rate": funding_rate,
        "apply_funding_schedule": apply_funding_schedule,
    }
    run_id = hashlib.sha256(repr(sorted(config.items())).encode()).hexdigest()[:24]
    trades = _trade_rows(run_id, cast(list[dict[str, Any]], result.trades.to_dict("records")))

    with session_scope(session_factory) as session:
        repo = PostgresBacktestRepository(session)
        repo.upsert_run(
            run_id=run_id,
            created_at=utc_now(),
            config_json=config,
            metrics_json=result.metrics,
        )
        repo.upsert_trades(trades)

    return BacktestPipelineResult(
        run_id=run_id,
        symbol=symbol,
        timeframe=timeframe,
        dataset_rows=len(dataset),
        signal_rows=len(signals),
        trade_rows=len(trades),
        metrics=result.metrics,
    )


def run_walk_forward_pipeline_from_postgres(
    session_factory: sessionmaker,
    symbol: str,
    timeframe: str,
    source: str | None = None,
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
) -> WalkForwardPipelineResult:
    with session_scope(session_factory) as session:
        market_repo = PostgresMarketDataRepository(session)
        bars = market_repo.fetch_raw_bars(symbol=symbol, timeframe=timeframe, source=source)

    if bars.empty:
        raise ValueError(f"No raw bars found for {symbol} {timeframe}.")

    dataset = build_dataset_from_bars(
        bars=bars,
        symbol=symbol,
        timeframe=timeframe,
        label_horizon=3,
    )
    signals = generate_mean_reversion_signals(dataset, symbol=symbol)
    walk_forward_result = run_walk_forward_backtest(
        signals=signals,
        bars=bars,
        costs=CostAssumptions(),
        initial_equity=initial_equity,
        position_fraction=position_fraction,
        max_holding_bars=max_holding_bars,
        max_concurrent_positions=max_concurrent_positions,
        max_positions_per_symbol=max_positions_per_symbol,
        cooldown_bars_after_stop=cooldown_bars_after_stop,
        allow_opposite_side_same_symbol=allow_opposite_side_same_symbol,
        funding_rate=funding_rate,
        apply_funding_schedule=apply_funding_schedule,
        n_folds=n_folds,
        min_train_bars=min_train_bars,
        test_bars=test_bars,
    )

    return WalkForwardPipelineResult(
        symbol=symbol,
        timeframe=timeframe,
        dataset_rows=len(dataset),
        signal_rows=len(signals),
        fold_rows=len(walk_forward_result.folds),
        aggregate_metrics=walk_forward_result.aggregate_metrics,
        folds=[fold.to_dict() for fold in walk_forward_result.folds],
    )


def run_ml_walk_forward_backtest_from_postgres(
    session_factory: sessionmaker,
    model_run_id: str,
    symbol: str,
    timeframe: str,
    feature_set_version: str = FEATURE_SET_LEGACY,
    source: str | None = None,
    artifact_base_dir: Path | str = Path("artifacts/models"),
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
    min_confidence: float = 0.55,
    min_edge_delta: float = 0.05,
    stop_loss_pct: float = 0.005,
    take_profit_pct: float = 0.010,
    ttl_sec: int = 1800,
) -> MlWalkForwardPipelineResult:
    bars, dataset = _load_bars_and_dataset(
        session_factory=session_factory,
        symbol=symbol,
        timeframe=timeframe,
        source=source,
        feature_set_version=feature_set_version,
    )

    probabilities = predict_advisor_probabilities(
        features=dataset,
        model_run_id=model_run_id,
        artifact_base_dir=artifact_base_dir,
    )
    signals = predictions_to_signal_frame(
        dataset=dataset,
        probabilities=probabilities,
        model_run_id=model_run_id,
        symbol_fallback=symbol,
        min_confidence=min_confidence,
        min_edge_delta=min_edge_delta,
        ttl_sec=ttl_sec,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )

    walk_forward_result = run_walk_forward_backtest(
        signals=signals,
        bars=bars,
        costs=CostAssumptions(),
        initial_equity=initial_equity,
        position_fraction=position_fraction,
        max_holding_bars=max_holding_bars,
        max_concurrent_positions=max_concurrent_positions,
        max_positions_per_symbol=max_positions_per_symbol,
        cooldown_bars_after_stop=cooldown_bars_after_stop,
        allow_opposite_side_same_symbol=allow_opposite_side_same_symbol,
        funding_rate=funding_rate,
        apply_funding_schedule=apply_funding_schedule,
        n_folds=n_folds,
        min_train_bars=min_train_bars,
        test_bars=test_bars,
    )

    traded_signal_count = int((signals["side"] != "no_trade").sum()) if not signals.empty else 0
    config: dict[str, object] = {
        "model_run_id": model_run_id,
        "feature_set_version": feature_set_version,
        "min_confidence": min_confidence,
        "min_edge_delta": min_edge_delta,
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
        "ttl_sec": ttl_sec,
        "n_folds": n_folds,
        "min_train_bars": min_train_bars,
        "test_bars": test_bars,
        "position_fraction": position_fraction,
        "max_holding_bars": max_holding_bars,
        "max_concurrent_positions": max_concurrent_positions,
        "max_positions_per_symbol": max_positions_per_symbol,
        "cooldown_bars_after_stop": cooldown_bars_after_stop,
        "allow_opposite_side_same_symbol": allow_opposite_side_same_symbol,
        "apply_funding_schedule": apply_funding_schedule,
        "funding_rate": funding_rate,
    }

    return MlWalkForwardPipelineResult(
        model_run_id=model_run_id,
        symbol=symbol,
        timeframe=timeframe,
        feature_set_version=feature_set_version,
        signal_count=len(signals),
        traded_signal_count=traded_signal_count,
        aggregate_metrics=walk_forward_result.aggregate_metrics,
        folds=[fold.to_dict() for fold in walk_forward_result.folds],
        config=config,
    )


def run_strategy_comparison_from_postgres(
    session_factory: sessionmaker,
    model_run_id: str,
    symbol: str,
    timeframe: str,
    feature_set_version: str = FEATURE_SET_LEGACY,
    source: str | None = None,
    artifact_base_dir: Path | str = Path("artifacts/models"),
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
    min_confidence: float = 0.55,
    min_edge_delta: float = 0.05,
    stop_loss_pct: float = 0.005,
    take_profit_pct: float = 0.010,
    ttl_sec: int = 1800,
) -> StrategyComparisonPipelineResult:
    bars, dataset = _load_bars_and_dataset(
        session_factory=session_factory,
        symbol=symbol,
        timeframe=timeframe,
        source=source,
        feature_set_version=feature_set_version,
    )
    assumptions_signature = hashlib.sha256(
        repr(
            sorted(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "feature_set_version": feature_set_version,
                    "initial_equity": initial_equity,
                    "position_fraction": position_fraction,
                    "max_holding_bars": max_holding_bars,
                    "max_concurrent_positions": max_concurrent_positions,
                    "max_positions_per_symbol": max_positions_per_symbol,
                    "cooldown_bars_after_stop": cooldown_bars_after_stop,
                    "allow_opposite_side_same_symbol": allow_opposite_side_same_symbol,
                    "funding_rate": funding_rate,
                    "apply_funding_schedule": apply_funding_schedule,
                    "n_folds": n_folds,
                    "min_train_bars": min_train_bars,
                    "test_bars": test_bars,
                }.items()
            )
        ).encode("utf-8")
    ).hexdigest()[:16]

    probabilities = predict_advisor_probabilities(
        features=dataset,
        model_run_id=model_run_id,
        artifact_base_dir=artifact_base_dir,
    )
    ml_signals = predictions_to_signal_frame(
        dataset=dataset,
        probabilities=probabilities,
        model_run_id=model_run_id,
        symbol_fallback=symbol,
        min_confidence=min_confidence,
        min_edge_delta=min_edge_delta,
        ttl_sec=ttl_sec,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )
    rule_signals = generate_mean_reversion_signals(dataset, symbol=symbol)

    strategy_inputs = [
        ("ml_advisor", ml_signals),
        ("mean_reversion", rule_signals),
    ]
    comparison_rows: list[dict[str, object]] = []

    for strategy_name, signals in strategy_inputs:
        wf_result = run_walk_forward_backtest(
            signals=signals,
            bars=bars,
            costs=CostAssumptions(),
            initial_equity=initial_equity,
            position_fraction=position_fraction,
            max_holding_bars=max_holding_bars,
            max_concurrent_positions=max_concurrent_positions,
            max_positions_per_symbol=max_positions_per_symbol,
            cooldown_bars_after_stop=cooldown_bars_after_stop,
            allow_opposite_side_same_symbol=allow_opposite_side_same_symbol,
            funding_rate=funding_rate,
            apply_funding_schedule=apply_funding_schedule,
            n_folds=n_folds,
            min_train_bars=min_train_bars,
            test_bars=test_bars,
        )
        comparison_rows.append(
            _build_strategy_row(
                strategy_name=strategy_name,
                signal_count=len(signals),
                assumptions_signature=assumptions_signature,
                wf_result=wf_result,
                initial_equity=initial_equity,
            )
        )

    sorted_by_return = sorted(
        comparison_rows,
        key=lambda row: _as_float(row.get("total_return", 0.0)),
        reverse=True,
    )
    for rank, row in enumerate(sorted_by_return, start=1):
        row["rank_by_total_return"] = rank
    sorted_by_pf = sorted(
        comparison_rows,
        key=lambda row: _as_float(row.get("profit_factor", 0.0)),
        reverse=True,
    )
    for rank, row in enumerate(sorted_by_pf, start=1):
        row["rank_by_profit_factor"] = rank
    sorted_by_drawdown = sorted(
        comparison_rows,
        key=lambda row: _as_float(row.get("max_drawdown", 0.0)),
        reverse=True,
    )
    for rank, row in enumerate(sorted_by_drawdown, start=1):
        row["rank_by_drawdown"] = rank

    ml_row = next(row for row in comparison_rows if row["strategy_name"] == "ml_advisor")
    rule_rows = [row for row in comparison_rows if row["strategy_name"] != "ml_advisor"]
    best_rule_by_return = max(rule_rows, key=lambda row: _as_float(row.get("total_return", 0.0)))
    best_rule_by_pf = max(rule_rows, key=lambda row: _as_float(row.get("profit_factor", 0.0)))
    best_by_return = max(comparison_rows, key=lambda row: _as_float(row.get("total_return", 0.0)))
    best_by_pf = max(comparison_rows, key=lambda row: _as_float(row.get("profit_factor", 0.0)))
    safest_by_drawdown = max(
        comparison_rows,
        key=lambda row: _as_float(row.get("max_drawdown", 0.0)),
    )
    ml_underperforms = _as_float(ml_row.get("total_return", 0.0)) < _as_float(
        best_rule_by_return.get("total_return", 0.0)
    )

    summary: dict[str, object] = {
        "best_strategy_by_total_return": str(best_by_return.get("strategy_name", "")),
        "best_strategy_by_profit_factor": str(best_by_pf.get("strategy_name", "")),
        "safest_strategy_by_drawdown": str(safest_by_drawdown.get("strategy_name", "")),
        "ml_vs_best_rule_total_return_delta": _as_float(ml_row.get("total_return", 0.0))
        - _as_float(best_rule_by_return.get("total_return", 0.0)),
        "ml_vs_best_rule_profit_factor_delta": _as_float(ml_row.get("profit_factor", 0.0))
        - _as_float(best_rule_by_pf.get("profit_factor", 0.0)),
        "warning": (
            f"ML underperforms best rule baseline ({best_rule_by_return.get('strategy_name', '')})"
            if ml_underperforms
            else ""
        ),
    }
    quality_gates, quality_gate_passed, quality_gate_warnings = _evaluate_quality_gates(
        comparison_rows=comparison_rows,
        summary=summary,
    )

    comparison_config: dict[str, object] = {
        "model_run_id": model_run_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "feature_set_version": feature_set_version,
        "n_folds": n_folds,
        "min_train_bars": min_train_bars,
        "test_bars": test_bars,
        "position_fraction": position_fraction,
        "max_holding_bars": max_holding_bars,
        "max_concurrent_positions": max_concurrent_positions,
        "max_positions_per_symbol": max_positions_per_symbol,
        "cooldown_bars_after_stop": cooldown_bars_after_stop,
        "allow_opposite_side_same_symbol": allow_opposite_side_same_symbol,
        "apply_funding_schedule": apply_funding_schedule,
        "funding_rate": funding_rate,
        "min_confidence": min_confidence,
        "min_edge_delta": min_edge_delta,
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
        "ttl_sec": ttl_sec,
        "assumptions_signature": assumptions_signature,
    }

    return StrategyComparisonPipelineResult(
        model_run_id=model_run_id,
        symbol=symbol,
        timeframe=timeframe,
        feature_set_version=feature_set_version,
        compared_strategies=[str(row.get("strategy_name", "")) for row in comparison_rows],
        comparison_rows=comparison_rows,
        summary=summary,
        quality_gates=quality_gates,
        quality_gate_passed=quality_gate_passed,
        quality_gate_warnings=quality_gate_warnings,
        comparison_config=comparison_config,
    )
