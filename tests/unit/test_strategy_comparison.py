from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from app.core.safety import LiveTradingDisabledError
from app.execution.real_disabled import RealBrokerInterface
from app.pipelines.backtest_pipeline import run_strategy_comparison_from_postgres


def _wf_stub(
    *,
    total_return: float,
    profit_factor: float,
    max_drawdown: float,
    final_equity: float,
    trade_count: float = 2.0,
    fold_returns: list[float] | None = None,
    fold_trade_counts: list[float] | None = None,
) -> SimpleNamespace:
    resolved_returns = fold_returns if fold_returns is not None else [total_return, total_return]
    resolved_trade_counts = (
        fold_trade_counts
        if fold_trade_counts is not None
        else [trade_count] * len(resolved_returns)
    )
    folds = []
    for fold_return, fold_trades in zip(resolved_returns, resolved_trade_counts, strict=True):
        folds.append(
            SimpleNamespace(
                metrics={
                    "total_return": fold_return,
                    "hit_rate": 0.5,
                    "profit_factor": profit_factor,
                    "max_drawdown": max_drawdown,
                    "final_equity": final_equity,
                    "trade_count": fold_trades,
                }
            )
        )
    positive_ratio = sum(1 for value in resolved_returns if value > 0.0) / len(resolved_returns)
    return SimpleNamespace(
        folds=folds,
        aggregate_metrics={"wf_positive_fold_ratio": positive_ratio},
    )


def test_comparison_includes_ml_and_mean_reversion(monkeypatch: pytest.MonkeyPatch) -> None:
    bars = pd.DataFrame(
        {
            "event_time": [datetime(2026, 1, 1, tzinfo=UTC)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
        }
    )
    dataset = pd.DataFrame(
        {"event_time": [datetime(2026, 1, 1, tzinfo=UTC)], "symbol": ["BTCUSDT"], "close": [100.0]}
    )
    ml_signals = pd.DataFrame(
        {
            "event_time": [datetime(2026, 1, 1, tzinfo=UTC)],
            "side": ["long"],
            "entry_ref": [100.0],
            "take_ref": [101.0],
            "stop_ref": [99.0],
            "symbol": ["BTCUSDT"],
        }
    )
    rule_signals = pd.DataFrame(
        {
            "event_time": [datetime(2026, 1, 1, tzinfo=UTC)],
            "side": ["short"],
            "entry_ref": [100.0],
            "take_ref": [99.0],
            "stop_ref": [101.0],
            "symbol": ["BTCUSDT"],
        }
    )

    monkeypatch.setattr(
        "app.pipelines.backtest_pipeline._load_bars_and_dataset",
        lambda **kwargs: (bars, dataset),
    )
    monkeypatch.setattr(
        "app.pipelines.backtest_pipeline.predict_advisor_probabilities",
        lambda **kwargs: pd.DataFrame(
            {
                "feature_ml_no_edge_probability": [0.2],
                "feature_ml_long_probability": [0.7],
                "feature_ml_short_probability": [0.1],
            }
        ),
    )
    monkeypatch.setattr(
        "app.pipelines.backtest_pipeline.predictions_to_signal_frame",
        lambda **kwargs: ml_signals,
    )
    monkeypatch.setattr(
        "app.pipelines.backtest_pipeline.generate_mean_reversion_signals",
        lambda features, symbol: rule_signals,
    )
    monkeypatch.setattr(
        "app.pipelines.backtest_pipeline.run_walk_forward_backtest",
        lambda **kwargs: _wf_stub(
            total_return=0.01 if kwargs["signals"] is ml_signals else 0.02,
            profit_factor=1.1 if kwargs["signals"] is ml_signals else 1.3,
            max_drawdown=-0.03 if kwargs["signals"] is ml_signals else -0.02,
            final_equity=10_100.0 if kwargs["signals"] is ml_signals else 10_200.0,
        ),
    )

    result = run_strategy_comparison_from_postgres(
        session_factory=None,  # type: ignore[arg-type]
        model_run_id="model1",
        symbol="BTCUSDT",
        timeframe="1m",
    )

    assert "ml_advisor" in result.compared_strategies
    assert "mean_reversion" in result.compared_strategies
    assert len(result.comparison_rows) == 2
    assert "quality_gates" in result.to_dict()
    assert "comparison_config" in result.to_dict()


def test_comparison_uses_same_assumptions_and_has_ranking_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars = pd.DataFrame(
        {
            "event_time": [datetime(2026, 1, 1, tzinfo=UTC)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
        }
    )
    dataset = pd.DataFrame(
        {"event_time": [datetime(2026, 1, 1, tzinfo=UTC)], "symbol": ["BTCUSDT"], "close": [100.0]}
    )
    signals = pd.DataFrame(
        {
            "event_time": [datetime(2026, 1, 1, tzinfo=UTC)],
            "side": ["long"],
            "entry_ref": [100.0],
            "take_ref": [101.0],
            "stop_ref": [99.0],
            "symbol": ["BTCUSDT"],
        }
    )

    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        "app.pipelines.backtest_pipeline._load_bars_and_dataset",
        lambda **kwargs: (bars, dataset),
    )
    monkeypatch.setattr(
        "app.pipelines.backtest_pipeline.predict_advisor_probabilities",
        lambda **kwargs: pd.DataFrame(
            {
                "feature_ml_no_edge_probability": [0.2],
                "feature_ml_long_probability": [0.7],
                "feature_ml_short_probability": [0.1],
            }
        ),
    )
    monkeypatch.setattr(
        "app.pipelines.backtest_pipeline.predictions_to_signal_frame",
        lambda **kwargs: signals,
    )
    monkeypatch.setattr(
        "app.pipelines.backtest_pipeline.generate_mean_reversion_signals",
        lambda features, symbol: signals,
    )

    def _capture(**kwargs):
        calls.append(kwargs)
        return _wf_stub(
            total_return=0.01,
            profit_factor=1.2,
            max_drawdown=-0.02,
            final_equity=10100.0,
        )

    monkeypatch.setattr("app.pipelines.backtest_pipeline.run_walk_forward_backtest", _capture)
    result = run_strategy_comparison_from_postgres(
        session_factory=None,  # type: ignore[arg-type]
        model_run_id="model2",
        symbol="BTCUSDT",
        timeframe="1m",
        n_folds=4,
        max_holding_bars=5,
        position_fraction=0.2,
    )

    assert len(calls) == 2
    assert calls[0]["n_folds"] == calls[1]["n_folds"] == 4
    assert calls[0]["max_holding_bars"] == calls[1]["max_holding_bars"] == 5
    assert calls[0]["position_fraction"] == calls[1]["position_fraction"] == 0.2
    for row in result.comparison_rows:
        assert "rank_by_total_return" in row
        assert "rank_by_profit_factor" in row
        assert "rank_by_drawdown" in row
        assert "sum_fold_total_return" in row
        assert "avg_fold_total_return" in row
        assert "median_fold_total_return" in row
        assert "worst_fold_total_return" in row
        assert "best_fold_total_return" in row
        assert "avg_fold_final_equity" in row
        assert "fold_count" in row
        assert "positive_fold_count" in row
        assert "negative_fold_count" in row
        assert "zero_trade_fold_count" in row
        assert "assumptions_signature" in row
        assert row["legacy_total_return_definition"] == "sum_fold_total_return"


def test_warning_when_ml_underperforms_rule_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    bars = pd.DataFrame(
        {
            "event_time": [datetime(2026, 1, 1, tzinfo=UTC)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
        }
    )
    dataset = pd.DataFrame(
        {"event_time": [datetime(2026, 1, 1, tzinfo=UTC)], "symbol": ["BTCUSDT"], "close": [100.0]}
    )
    ml_signals = pd.DataFrame(
        {
            "event_time": [datetime(2026, 1, 1, tzinfo=UTC)],
            "side": ["long"],
            "entry_ref": [100.0],
            "take_ref": [101.0],
            "stop_ref": [99.0],
            "symbol": ["BTCUSDT"],
        }
    )
    rule_signals = ml_signals.copy()

    monkeypatch.setattr(
        "app.pipelines.backtest_pipeline._load_bars_and_dataset",
        lambda **kwargs: (bars, dataset),
    )
    monkeypatch.setattr(
        "app.pipelines.backtest_pipeline.predict_advisor_probabilities",
        lambda **kwargs: pd.DataFrame(
            {
                "feature_ml_no_edge_probability": [0.2],
                "feature_ml_long_probability": [0.7],
                "feature_ml_short_probability": [0.1],
            }
        ),
    )
    monkeypatch.setattr(
        "app.pipelines.backtest_pipeline.predictions_to_signal_frame",
        lambda **kwargs: ml_signals,
    )
    monkeypatch.setattr(
        "app.pipelines.backtest_pipeline.generate_mean_reversion_signals",
        lambda features, symbol: rule_signals,
    )
    monkeypatch.setattr(
        "app.pipelines.backtest_pipeline.run_walk_forward_backtest",
        lambda **kwargs: _wf_stub(
            total_return=0.01 if kwargs["signals"] is ml_signals else 0.03,
            profit_factor=1.1 if kwargs["signals"] is ml_signals else 1.6,
            max_drawdown=-0.03 if kwargs["signals"] is ml_signals else -0.02,
            final_equity=10100.0 if kwargs["signals"] is ml_signals else 10300.0,
        ),
    )

    result = run_strategy_comparison_from_postgres(
        session_factory=None,  # type: ignore[arg-type]
        model_run_id="model3",
        symbol="BTCUSDT",
        timeframe="1m",
    )
    assert "underperforms" in str(result.summary["warning"])
    assert any(
        "does not beat best rule baseline" in warning for warning in result.quality_gate_warnings
    )


def test_fold_aggregate_fields_are_computed_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    bars = pd.DataFrame(
        {
            "event_time": [datetime(2026, 1, 1, tzinfo=UTC)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
        }
    )
    dataset = pd.DataFrame(
        {"event_time": [datetime(2026, 1, 1, tzinfo=UTC)], "symbol": ["BTCUSDT"], "close": [100.0]}
    )
    signals = pd.DataFrame(
        {
            "event_time": [datetime(2026, 1, 1, tzinfo=UTC)],
            "side": ["long"],
            "entry_ref": [100.0],
            "take_ref": [101.0],
            "stop_ref": [99.0],
            "symbol": ["BTCUSDT"],
        }
    )

    monkeypatch.setattr(
        "app.pipelines.backtest_pipeline._load_bars_and_dataset",
        lambda **kwargs: (bars, dataset),
    )
    monkeypatch.setattr(
        "app.pipelines.backtest_pipeline.predict_advisor_probabilities",
        lambda **kwargs: pd.DataFrame(
            {
                "feature_ml_no_edge_probability": [0.2],
                "feature_ml_long_probability": [0.7],
                "feature_ml_short_probability": [0.1],
            }
        ),
    )
    monkeypatch.setattr(
        "app.pipelines.backtest_pipeline.predictions_to_signal_frame",
        lambda **kwargs: signals,
    )
    monkeypatch.setattr(
        "app.pipelines.backtest_pipeline.generate_mean_reversion_signals",
        lambda features, symbol: signals,
    )

    def _capture(**kwargs):
        if kwargs["signals"] is signals:
            return _wf_stub(
                total_return=0.0,
                profit_factor=1.2,
                max_drawdown=-0.03,
                final_equity=10050.0,
                fold_returns=[0.10, -0.05, 0.00],
                fold_trade_counts=[2.0, 1.0, 0.0],
            )
        return _wf_stub(
            total_return=0.0,
            profit_factor=1.1,
            max_drawdown=-0.02,
            final_equity=10080.0,
            fold_returns=[0.02, 0.01, -0.01],
            fold_trade_counts=[1.0, 1.0, 1.0],
        )

    monkeypatch.setattr("app.pipelines.backtest_pipeline.run_walk_forward_backtest", _capture)
    result = run_strategy_comparison_from_postgres(
        session_factory=None,  # type: ignore[arg-type]
        model_run_id="model4",
        symbol="BTCUSDT",
        timeframe="1m",
    )

    ml_row = next(row for row in result.comparison_rows if row["strategy_name"] == "ml_advisor")
    assert ml_row["sum_fold_total_return"] == pytest.approx(0.05)
    assert ml_row["avg_fold_total_return"] == pytest.approx(0.05 / 3)
    assert ml_row["median_fold_total_return"] == pytest.approx(0.0)
    assert ml_row["worst_fold_total_return"] == pytest.approx(-0.05)
    assert ml_row["best_fold_total_return"] == pytest.approx(0.10)
    assert ml_row["zero_trade_fold_count"] == 1


def test_quality_gate_fails_when_ml_has_no_trades(monkeypatch: pytest.MonkeyPatch) -> None:
    bars = pd.DataFrame(
        {
            "event_time": [datetime(2026, 1, 1, tzinfo=UTC)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
        }
    )
    dataset = pd.DataFrame(
        {"event_time": [datetime(2026, 1, 1, tzinfo=UTC)], "symbol": ["BTCUSDT"], "close": [100.0]}
    )
    signals = pd.DataFrame(
        {
            "event_time": [datetime(2026, 1, 1, tzinfo=UTC)],
            "side": ["long"],
            "entry_ref": [100.0],
            "take_ref": [101.0],
            "stop_ref": [99.0],
            "symbol": ["BTCUSDT"],
        }
    )
    monkeypatch.setattr(
        "app.pipelines.backtest_pipeline._load_bars_and_dataset",
        lambda **kwargs: (bars, dataset),
    )
    monkeypatch.setattr(
        "app.pipelines.backtest_pipeline.predict_advisor_probabilities",
        lambda **kwargs: pd.DataFrame(
            {
                "feature_ml_no_edge_probability": [0.2],
                "feature_ml_long_probability": [0.7],
                "feature_ml_short_probability": [0.1],
            }
        ),
    )
    monkeypatch.setattr(
        "app.pipelines.backtest_pipeline.predictions_to_signal_frame",
        lambda **kwargs: signals,
    )
    monkeypatch.setattr(
        "app.pipelines.backtest_pipeline.generate_mean_reversion_signals",
        lambda features, symbol: signals,
    )

    def _capture(**kwargs):
        if kwargs["signals"] is signals:
            return _wf_stub(
                total_return=0.0,
                profit_factor=0.0,
                max_drawdown=0.0,
                final_equity=10000.0,
                fold_returns=[0.0, 0.0, 0.0],
                fold_trade_counts=[0.0, 0.0, 0.0],
            )
        return _wf_stub(
            total_return=0.01,
            profit_factor=1.1,
            max_drawdown=-0.02,
            final_equity=10020.0,
            fold_returns=[0.01, 0.01, 0.01],
            fold_trade_counts=[1.0, 1.0, 1.0],
        )

    monkeypatch.setattr("app.pipelines.backtest_pipeline.run_walk_forward_backtest", _capture)
    result = run_strategy_comparison_from_postgres(
        session_factory=None,  # type: ignore[arg-type]
        model_run_id="model5",
        symbol="BTCUSDT",
        timeframe="1m",
    )
    assert result.quality_gates["ml_has_trades"] is False
    assert result.quality_gate_passed is False
    assert any("produced no trades" in warning for warning in result.quality_gate_warnings)


def test_comparison_config_contains_assumption_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    bars = pd.DataFrame(
        {
            "event_time": [datetime(2026, 1, 1, tzinfo=UTC)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
        }
    )
    dataset = pd.DataFrame(
        {"event_time": [datetime(2026, 1, 1, tzinfo=UTC)], "symbol": ["BTCUSDT"], "close": [100.0]}
    )
    signals = pd.DataFrame(
        {
            "event_time": [datetime(2026, 1, 1, tzinfo=UTC)],
            "side": ["long"],
            "entry_ref": [100.0],
            "take_ref": [101.0],
            "stop_ref": [99.0],
            "symbol": ["BTCUSDT"],
        }
    )
    monkeypatch.setattr(
        "app.pipelines.backtest_pipeline._load_bars_and_dataset",
        lambda **kwargs: (bars, dataset),
    )
    monkeypatch.setattr(
        "app.pipelines.backtest_pipeline.predict_advisor_probabilities",
        lambda **kwargs: pd.DataFrame(
            {
                "feature_ml_no_edge_probability": [0.2],
                "feature_ml_long_probability": [0.7],
                "feature_ml_short_probability": [0.1],
            }
        ),
    )
    monkeypatch.setattr(
        "app.pipelines.backtest_pipeline.predictions_to_signal_frame",
        lambda **kwargs: signals,
    )
    monkeypatch.setattr(
        "app.pipelines.backtest_pipeline.generate_mean_reversion_signals",
        lambda features, symbol: signals,
    )
    monkeypatch.setattr(
        "app.pipelines.backtest_pipeline.run_walk_forward_backtest",
        lambda **kwargs: _wf_stub(
            total_return=0.01,
            profit_factor=1.1,
            max_drawdown=-0.03,
            final_equity=10100.0,
            fold_returns=[0.01, 0.02, -0.01],
            fold_trade_counts=[1.0, 1.0, 1.0],
        ),
    )
    result = run_strategy_comparison_from_postgres(
        session_factory=None,  # type: ignore[arg-type]
        model_run_id="model6",
        symbol="BTCUSDT",
        timeframe="1m",
        n_folds=3,
        min_train_bars=120,
        test_bars=20,
        position_fraction=0.2,
        max_holding_bars=4,
    )
    config = result.comparison_config
    assert "assumptions_signature" in config
    assert config["n_folds"] == 3
    assert config["position_fraction"] == 0.2
    assert config["max_holding_bars"] == 4


def test_live_execution_remains_disabled() -> None:
    broker = RealBrokerInterface()
    with pytest.raises(LiveTradingDisabledError):
        broker.place_order({"symbol": "BTCUSDT", "side": "Buy", "qty": 0.001})
