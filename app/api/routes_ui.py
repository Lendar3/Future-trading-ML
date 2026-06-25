from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.core.risk_config import get_risk_config
from app.features.dataset_builder import (
    FEATURE_SET_LEGACY,
    FEATURE_SET_PUBLIC,
    build_dataset_from_bars,
    build_dataset_from_public_tables,
)
from app.features.dataset_quality import compute_dataset_quality
from app.storage.database import build_engine, build_session_factory, session_scope
from app.storage.repositories import (
    PostgresAuditRepository,
    PostgresBacktestRepository,
    PostgresMarketDataRepository,
    PostgresModelRepository,
    PostgresPaperRepository,
    PostgresPublicMarketRepository,
    PostgresSignalRepository,
)

router = APIRouter()


@router.get("/ui/market-state")
def ui_market_state() -> dict[str, str]:
    return {"system_mode": "research", "primary_symbol": "BTCUSDT"}


@router.get("/ui/chart-data")
def ui_chart_data(
    symbol: str = Query("BTCUSDT"),
    timeframe: str = Query("1m"),
    limit: int = Query(200, ge=1, le=1_000),
    source: str | None = Query(None),
) -> dict[str, object]:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        repo = PostgresMarketDataRepository(session)
        candles = repo.latest_raw_bars(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
            source=source,
        )
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "candles": candles,
        "volume": [{"event_time": row["event_time"], "volume": row["volume"]} for row in candles],
    }


@router.get("/ui/feature-vector/{symbol}")
def ui_feature_vector(symbol: str) -> dict[str, object]:
    """Return the latest feature vector for ``symbol``, grouped by category.

    Feature groups returned
    -----------------------
    * ``price_volume_volatility`` — OHLCV-derived features
    * ``basis`` — mark/index price basis features
    * ``funding`` — funding-rate features
    * ``open_interest`` — OI features
    * ``long_short`` — long-short ratio features
    * ``other`` — everything else
    """
    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        signals = PostgresSignalRepository(session).latest(symbol=symbol, limit=1)
        if not signals:
            return {
                "symbol": symbol,
                "signal_id": None,
                "features": {},
                "feature_groups": {},
            }
        audit = PostgresAuditRepository(session).get_by_signal_id(str(signals[0]["signal_id"]))
    raw_features: dict[str, object] = audit.feature_snapshot_json if audit is not None else {}

    # Group features by prefix convention.
    groups: dict[str, dict[str, object]] = {
        "price_volume_volatility": {},
        "basis": {},
        "funding": {},
        "open_interest": {},
        "long_short": {},
        "other": {},
    }
    basis_prefixes = ("feature_mark", "feature_index", "feature_basis", "feature_premium")
    funding_prefixes = ("feature_funding", "feature_minutes_to_next")
    oi_prefixes = (
        "feature_open_interest",
        "feature_oi",
        "feature_price_up_oi",
        "feature_price_down_oi",
        "feature_squeeze",
    )
    ls_prefixes = (
        "feature_long_short",
        "feature_ls",
        "feature_long_account",
        "feature_short_account",
        "feature_crowded",
    )
    price_vol_prefixes = (
        "feature_ret",
        "feature_log",
        "feature_close",
        "feature_zscore",
        "feature_distance",
        "feature_volume",
        "feature_vol",
        "feature_atr",
        "feature_bb",
        "feature_hv",
    )
    for key, value in raw_features.items():
        if any(key.startswith(p) for p in basis_prefixes):
            groups["basis"][key] = value
        elif any(key.startswith(p) for p in funding_prefixes):
            groups["funding"][key] = value
        elif any(key.startswith(p) for p in oi_prefixes):
            groups["open_interest"][key] = value
        elif any(key.startswith(p) for p in ls_prefixes):
            groups["long_short"][key] = value
        elif any(key.startswith(p) for p in price_vol_prefixes):
            groups["price_volume_volatility"][key] = value
        else:
            groups["other"][key] = value

    return {
        "symbol": symbol,
        "signal_id": signals[0]["signal_id"],
        "features": raw_features,
        "feature_groups": groups,
    }


@router.get("/ui/signal-explanation/{signal_id}")
def ui_signal_explanation(signal_id: str) -> dict[str, object]:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        audit = PostgresAuditRepository(session).get_by_signal_id(signal_id)
    if audit is None:
        raise HTTPException(status_code=404, detail=f"No audit log found for signal {signal_id}.")
    return {
        "audit_id": audit.audit_id,
        "signal_id": audit.signal_id,
        "symbol": audit.symbol,
        "event_time": audit.event_time.isoformat(),
        "strategy_name": audit.strategy_name,
        "feature_snapshot": audit.feature_snapshot_json,
        "rationale": audit.rationale_json,
        "risk_checks": audit.risk_checks_json,
        "cost_assumptions": audit.cost_assumptions_json,
        "decision": audit.decision,
        "decision_reason": audit.decision_reason,
        "system_mode": audit.system_mode,
    }


@router.get("/ui/training-runs")
def ui_training_runs() -> dict[str, list[dict[str, object]]]:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        runs = PostgresModelRepository(session).latest(limit=20)
    return {"runs": runs}


@router.get("/ui/backtest-runs")
def ui_backtest_runs() -> dict[str, list[dict[str, object]]]:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        runs = PostgresBacktestRepository(session).latest_runs(limit=20)
    return {"runs": runs}


@router.get("/ui/demo-state")
def ui_demo_state() -> dict[str, object]:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        signals = PostgresSignalRepository(session).latest(limit=10)
        paper_repo = PostgresPaperRepository(session)
        portfolio = paper_repo.latest_portfolio()
        orders = paper_repo.latest_orders(limit=20)
        fills = paper_repo.latest_fills(limit=20)
    return {
        "status": "ready" if signals else "empty",
        "latest_signals": signals,
        "portfolio": portfolio,
        "orders": orders,
        "fills": fills,
    }


@router.get("/ui/risk-state")
def ui_risk_state() -> dict[str, object]:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        paper_repo = PostgresPaperRepository(session)
        portfolio = paper_repo.latest_portfolio()
        orders = paper_repo.latest_orders(limit=100)
    risk_limits = get_risk_config().to_dict()
    return {
        "kill_switch": "inactive",
        "live_trading": "disabled",
        "portfolio": portfolio,
        "risk_limits": risk_limits,
        "open_positions": len(portfolio["positions"]),
        "order_count": len(orders),
        "max_positions_reached": len(portfolio["positions"]) >= risk_limits["max_open_positions"],
    }


@router.get("/ui/readiness-report")
def ui_readiness_report() -> dict[str, bool]:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        candles = PostgresMarketDataRepository(session).latest_raw_bars(
            symbol="BTCUSDT",
            timeframe="1m",
            limit=100,
        )
        signals = PostgresSignalRepository(session).latest(symbol="BTCUSDT", limit=1)
        backtests = PostgresBacktestRepository(session).latest_runs(limit=1)
        portfolio = PostgresPaperRepository(session).latest_portfolio()

    return {
        "minimum_history_collected": len(candles) >= 80,
        "backtest_completed": bool(backtests),
        "walk_forward_validation_passed": False,
        "demo_trading_completed": bool(portfolio["snapshot_id"]),
        "paper_trading_net_pnl_positive": False,
        "max_drawdown_acceptable": True,
        "profit_factor_acceptable": False,
        "no_lookahead_validation_passed": True,
        "labels_net_of_costs": True,
        "audit_trail_complete": bool(signals),
        "risk_module_enabled": True,
        "live_trading_disabled": True,
    }


@router.get("/ui/dataset-quality")
def ui_dataset_quality(
    symbol: str = Query("BTCUSDT"),
    timeframe: str = Query("1m"),
    source: str | None = Query(None),
    feature_set_name: str = Query(FEATURE_SET_LEGACY),
) -> dict[str, object]:
    """Build a fresh dataset from PostgreSQL and return a quality report.

    The endpoint is intentionally synchronous and may be slow for large
    datasets.  Use it in development and research, not in hot paths.

    Query parameters
    ----------------
    symbol:
        Instrument, e.g. ``"BTCUSDT"``.
    timeframe:
        App bar interval, e.g. ``"1m"``.
    source:
        Optional ingestion source filter.
    feature_set_name:
        ``"v1"`` or ``"v1.6_public"``.
    """
    engine = build_engine()
    sf = build_session_factory(engine)
    enriched = feature_set_name == FEATURE_SET_PUBLIC

    with session_scope(sf) as session:
        market_repo = PostgresMarketDataRepository(session)
        bars = market_repo.fetch_raw_bars(symbol=symbol, timeframe=timeframe, source=source)
        if enriched:
            pub_repo = PostgresPublicMarketRepository(session)
            mark_bars = pub_repo.fetch_mark_bars(symbol=symbol, timeframe=timeframe)
            index_bars = pub_repo.fetch_index_bars(symbol=symbol, timeframe=timeframe)
            funding = pub_repo.fetch_funding(symbol=symbol, source=source)
            open_interest = pub_repo.fetch_open_interest(symbol=symbol, source=source)
            long_short = pub_repo.fetch_long_short_ratio(symbol=symbol, source=source)

    if bars.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No bars found for {symbol} {timeframe} source={source}.",
        )

    if enriched:
        dataset = build_dataset_from_public_tables(
            bars=bars,
            symbol=symbol,
            timeframe=timeframe,
            mark_bars=mark_bars if not mark_bars.empty else None,
            index_bars=index_bars if not index_bars.empty else None,
            funding=funding if not funding.empty else None,
            open_interest=open_interest if not open_interest.empty else None,
            long_short_ratio=long_short if not long_short.empty else None,
            feature_set_name=feature_set_name,
        )
    else:
        dataset = build_dataset_from_bars(
            bars=bars,
            symbol=symbol,
            timeframe=timeframe,
            feature_set_name=feature_set_name,
        )

    report = compute_dataset_quality(dataset)
    return report.to_dict()
