from __future__ import annotations

import pandas as pd
import streamlit as st

from app.dashboard.api_client import DashboardApiError, get_json
from app.dashboard.components.charts import candlestick_figure, volume_figure

st.set_page_config(page_title="Bybit Futures Edge Lab", layout="wide")
st.title("Bybit Futures Edge Lab")
st.caption(
    "Research, backtest, paper/demo, and advisory dashboard. Live trading is disabled in v1."
)

symbol = st.selectbox("Symbol", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h"])

try:
    market_state = get_json("/ui/market-state")
    demo_state = get_json("/ui/demo-state")
    chart_data = get_json(
        "/ui/chart-data",
        {"symbol": symbol, "timeframe": timeframe, "limit": 160},
    )
    signals = demo_state.get("latest_signals", [])
except DashboardApiError as exc:
    st.warning(str(exc))
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Mode", str(market_state.get("system_mode", "research")))
col2.metric("Live trading", "disabled")
col3.metric("Signals", str(len(signals)))
col4.metric("Demo data", str(demo_state.get("status", "empty")))

st.subheader("Market Overview")
candles = pd.DataFrame(chart_data.get("candles", []))
if candles.empty:
    st.info("No candles found. Run `make seed-demo` or POST /demo/seed first.")
else:
    st.plotly_chart(candlestick_figure(candles), use_container_width=True)
    st.plotly_chart(volume_figure(candles), use_container_width=True)

st.subheader("Latest Signals")
if signals:
    st.dataframe(pd.DataFrame(signals), use_container_width=True)
else:
    st.info("No signals persisted yet.")
