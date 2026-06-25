from __future__ import annotations

import pandas as pd
import streamlit as st

from app.dashboard.api_client import DashboardApiError, get_json
from app.dashboard.components.charts import candlestick_figure, volume_figure

st.title("Market Overview")
symbol = st.selectbox("Symbol", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h"])

try:
    chart_data = get_json(
        "/ui/chart-data",
        {"symbol": symbol, "timeframe": timeframe, "limit": 240},
    )
    signals = get_json("/signals/latest", {"symbol": symbol, "limit": 10})["signals"]
    feature_data = get_json(f"/ui/feature-vector/{symbol}")
except DashboardApiError as exc:
    st.warning(str(exc))
    st.stop()

candles = pd.DataFrame(chart_data.get("candles", []))
if candles.empty:
    st.info("No candles found. Run `make seed-demo` or `make backfill-public` first.")
else:
    st.plotly_chart(candlestick_figure(candles), use_container_width=True)
    st.plotly_chart(volume_figure(candles), use_container_width=True)

st.subheader("Latest Signals")
if signals:
    st.dataframe(pd.DataFrame(signals), use_container_width=True)
else:
    st.info("No signals found for this symbol.")

# ---- Feature Panel (grouped) -----------------------------------------
st.subheader("Feature Vector")
feature_groups: dict = feature_data.get("feature_groups", {})
all_features: dict = feature_data.get("features", {})

if not all_features:
    st.info(
        "No features yet. Run `make seed-demo` or `make backfill-public` "
        "then trigger a pipeline run."
    )
else:
    group_labels = {
        "price_volume_volatility": "📈 Price / Volume / Volatility",
        "basis": "⚖️ Basis (Mark − Index)",
        "funding": "💸 Funding",
        "open_interest": "📦 Open Interest",
        "long_short": "↕️ Long-Short Ratio",
        "other": "🔧 Other",
    }

    for group_key, group_title in group_labels.items():
        group_features = feature_groups.get(group_key, {})
        if not group_features:
            continue
        with st.expander(group_title, expanded=(group_key == "price_volume_volatility")):
            rows = [
                {"feature": k, "value": round(float(v), 6) if isinstance(v, (int, float)) else v}
                for k, v in sorted(group_features.items())
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
