from __future__ import annotations

import hashlib
import json

from fastapi import APIRouter, HTTPException, Query

from app.core.clock import utc_now
from app.pipelines.backtest_pipeline import (
    StrategyComparisonPipelineResult,
    run_backtest_pipeline_from_postgres,
    run_ml_walk_forward_backtest_from_postgres,
    run_strategy_comparison_from_postgres,
    run_walk_forward_pipeline_from_postgres,
)
from app.storage.database import build_engine, build_session_factory, session_scope
from app.storage.repositories import PostgresBacktestRepository, PostgresModelRepository

router = APIRouter()


def _strategy_snapshot_payload(result: StrategyComparisonPipelineResult) -> dict[str, object]:
    return {
        "model_run_id": result.model_run_id,
        "symbol": result.symbol,
        "timeframe": result.timeframe,
        "feature_set_version": result.feature_set_version,
        "compared_strategies": result.compared_strategies,
        "comparison_rows": result.comparison_rows,
        "summary": result.summary,
        "quality_gates": result.quality_gates,
        "quality_gate_passed": result.quality_gate_passed,
        "quality_gate_warnings": result.quality_gate_warnings,
        "comparison_config": result.comparison_config,
    }


def _strategy_snapshot_id(
    *,
    model_run_id: str,
    symbol: str,
    timeframe: str,
    feature_set_version: str,
    comparison_payload: dict[str, object],
) -> str:
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "model_run_id": model_run_id,
                "symbol": symbol,
                "timeframe": timeframe,
                "feature_set_version": feature_set_version,
                "comparison_payload": comparison_payload,
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:24]
    return f"scs_{fingerprint}"


@router.post("/backtest/run")
def run_backtest(
    symbol: str = Query("BTCUSDT"),
    timeframe: str = Query("1m"),
    source: str | None = Query(None),
    initial_equity: float = Query(10_000.0, gt=0),
    position_fraction: float = Query(0.10, gt=0, le=1.0),
    max_holding_bars: int = Query(3, ge=1, le=500),
    max_concurrent_positions: int = Query(1, ge=1, le=100),
    max_positions_per_symbol: int = Query(1, ge=1, le=100),
    cooldown_bars_after_stop: int = Query(0, ge=0, le=10_000),
    allow_opposite_side_same_symbol: bool = Query(False),
    funding_rate: float | None = Query(None, ge=-1.0, le=1.0),
    apply_funding_schedule: bool = Query(False),
) -> dict[str, object]:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    try:
        result = run_backtest_pipeline_from_postgres(
            session_factory=session_factory,
            symbol=symbol,
            timeframe=timeframe,
            source=source,
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
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"status": "ok", **result.to_dict()}


@router.get("/backtests/{run_id}")
def get_backtest(run_id: str) -> dict[str, object]:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        result = PostgresBacktestRepository(session).get_run(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Backtest {run_id} not found.")
    return result


@router.post("/backtest/walk-forward")
def run_backtest_walk_forward(
    symbol: str = Query("BTCUSDT"),
    timeframe: str = Query("1m"),
    source: str | None = Query(None),
    initial_equity: float = Query(10_000.0, gt=0),
    position_fraction: float = Query(0.10, gt=0, le=1.0),
    max_holding_bars: int = Query(3, ge=1, le=500),
    max_concurrent_positions: int = Query(1, ge=1, le=100),
    max_positions_per_symbol: int = Query(1, ge=1, le=100),
    cooldown_bars_after_stop: int = Query(0, ge=0, le=10_000),
    allow_opposite_side_same_symbol: bool = Query(False),
    funding_rate: float | None = Query(None, ge=-1.0, le=1.0),
    apply_funding_schedule: bool = Query(False),
    n_folds: int = Query(3, ge=1, le=50),
    min_train_bars: int = Query(100, ge=1, le=1_000_000),
    test_bars: int | None = Query(None, ge=1, le=1_000_000),
) -> dict[str, object]:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    try:
        result = run_walk_forward_pipeline_from_postgres(
            session_factory=session_factory,
            symbol=symbol,
            timeframe=timeframe,
            source=source,
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
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if message.startswith("No raw bars found") else 400
        raise HTTPException(status_code=status_code, detail=message) from exc
    return {"status": "ok", **result.to_dict()}


@router.post("/backtest/ml-walk-forward")
def run_ml_backtest_walk_forward(
    model_run_id: str = Query(..., min_length=4),
    symbol: str = Query("BTCUSDT"),
    timeframe: str = Query("1m"),
    feature_set_version: str = Query("v1"),
    source: str | None = Query(None),
    initial_equity: float = Query(10_000.0, gt=0),
    position_fraction: float = Query(0.10, gt=0, le=1.0),
    max_holding_bars: int = Query(3, ge=1, le=500),
    max_concurrent_positions: int = Query(1, ge=1, le=100),
    max_positions_per_symbol: int = Query(1, ge=1, le=100),
    cooldown_bars_after_stop: int = Query(0, ge=0, le=10_000),
    allow_opposite_side_same_symbol: bool = Query(False),
    apply_funding_schedule: bool = Query(False),
    funding_rate: float | None = Query(None, ge=-1.0, le=1.0),
    n_folds: int = Query(3, ge=1, le=100),
    min_train_bars: int = Query(100, ge=1, le=1_000_000),
    test_bars: int | None = Query(None, ge=1, le=1_000_000),
    min_confidence: float = Query(0.55, ge=0.0, le=1.0),
    min_edge_delta: float = Query(0.05, ge=0.0, le=1.0),
    stop_loss_pct: float = Query(0.005, ge=0.0, le=1.0),
    take_profit_pct: float = Query(0.010, ge=0.0, le=1.0),
    ttl_sec: int = Query(1800, ge=1, le=86_400),
) -> dict[str, object]:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    try:
        result = run_ml_walk_forward_backtest_from_postgres(
            session_factory=session_factory,
            model_run_id=model_run_id,
            symbol=symbol,
            timeframe=timeframe,
            feature_set_version=feature_set_version,
            source=source,
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
            min_confidence=min_confidence,
            min_edge_delta=min_edge_delta,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            ttl_sec=ttl_sec,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if message.startswith("No raw bars found") else 400
        raise HTTPException(status_code=status_code, detail=message) from exc
    return {"status": "ok", **result.to_dict()}


@router.post("/backtest/strategy-comparison")
def run_strategy_comparison(
    model_run_id: str = Query(..., min_length=4),
    symbol: str = Query("BTCUSDT"),
    timeframe: str = Query("1m"),
    feature_set_version: str = Query("v1"),
    source: str | None = Query(None),
    initial_equity: float = Query(10_000.0, gt=0),
    position_fraction: float = Query(0.10, gt=0, le=1.0),
    max_holding_bars: int = Query(3, ge=1, le=500),
    max_concurrent_positions: int = Query(1, ge=1, le=100),
    max_positions_per_symbol: int = Query(1, ge=1, le=100),
    cooldown_bars_after_stop: int = Query(0, ge=0, le=10_000),
    allow_opposite_side_same_symbol: bool = Query(False),
    apply_funding_schedule: bool = Query(False),
    funding_rate: float | None = Query(None, ge=-1.0, le=1.0),
    n_folds: int = Query(3, ge=1, le=100),
    min_train_bars: int = Query(100, ge=1, le=1_000_000),
    test_bars: int | None = Query(None, ge=1, le=1_000_000),
    min_confidence: float = Query(0.55, ge=0.0, le=1.0),
    min_edge_delta: float = Query(0.05, ge=0.0, le=1.0),
    stop_loss_pct: float = Query(0.005, ge=0.0, le=1.0),
    take_profit_pct: float = Query(0.010, ge=0.0, le=1.0),
    ttl_sec: int = Query(1800, ge=1, le=86_400),
) -> dict[str, object]:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    try:
        result = run_strategy_comparison_from_postgres(
            session_factory=session_factory,
            model_run_id=model_run_id,
            symbol=symbol,
            timeframe=timeframe,
            feature_set_version=feature_set_version,
            source=source,
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
            min_confidence=min_confidence,
            min_edge_delta=min_edge_delta,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            ttl_sec=ttl_sec,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if message.startswith("No raw bars found") else 400
        raise HTTPException(status_code=status_code, detail=message) from exc

    comparison_payload = _strategy_snapshot_payload(result)
    snapshot_id = _strategy_snapshot_id(
        model_run_id=result.model_run_id,
        symbol=result.symbol,
        timeframe=result.timeframe,
        feature_set_version=result.feature_set_version,
        comparison_payload=comparison_payload,
    )
    with session_scope(session_factory) as session:
        PostgresModelRepository(session).upsert_strategy_comparison_snapshot(
            snapshot_id=snapshot_id,
            model_run_id=result.model_run_id,
            symbol=result.symbol,
            timeframe=result.timeframe,
            feature_set_version=result.feature_set_version,
            created_at=utc_now(),
            comparison_json=comparison_payload,
        )
    return {"status": "ok", **result.to_dict()}
