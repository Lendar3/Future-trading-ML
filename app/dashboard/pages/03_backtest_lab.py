from __future__ import annotations

import pandas as pd
import streamlit as st

from app.dashboard.api_client import DashboardApiError, get_json, post_json

st.title("Backtest Lab")
symbol = st.selectbox("Symbol", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h"])
source = st.text_input("Source", "demo-seed")
strategy = st.selectbox(
    "Strategy",
    [
        "mean_reversion",
        "funding_arbitrage",
        "orderbook_imbalance",
        "cross_market_momentum",
        "advisor_ml",
    ],
)
st.selectbox("Slippage mode", ["conservative", "mixed_maker_taker", "taker_only"])
initial_equity = st.number_input("Initial equity", min_value=100.0, value=10_000.0, step=100.0)

if strategy != "mean_reversion":
    st.info("Only mean_reversion is wired to the current scaffold backtest engine.")

if st.button("Run Backtest"):
    try:
        result = post_json(
            "/backtest/run",
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "source": source or None,
                "initial_equity": initial_equity,
            },
        )
        st.success(f"Backtest run {result['run_id']} saved.")
        st.json(result["metrics"])
    except DashboardApiError as exc:
        st.warning(str(exc))

try:
    runs = get_json("/ui/backtest-runs")["runs"]
except DashboardApiError as exc:
    st.warning(str(exc))
    st.stop()

st.subheader("Recent Backtests")
if runs:
    rows = []
    for run in runs:
        metrics = run["metrics"]
        rows.append(
            {
                "run_id": run["run_id"],
                "created_at": run["created_at"],
                "symbol": run["config"].get("symbol"),
                "strategy": run["config"].get("strategy"),
                "trade_count": metrics.get("trade_count"),
                "total_return": metrics.get("total_return"),
                "hit_rate": metrics.get("hit_rate"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.info("No backtests persisted yet.")
