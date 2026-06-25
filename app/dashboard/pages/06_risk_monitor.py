from __future__ import annotations

import pandas as pd
import streamlit as st

from app.dashboard.api_client import DashboardApiError, get_json

st.title("Risk Monitor")

try:
    risk_state = get_json("/ui/risk-state")
except DashboardApiError as exc:
    st.warning(str(exc))
    st.stop()

portfolio = risk_state["portfolio"]
risk_limits = risk_state["risk_limits"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Kill-switch", str(risk_state["kill_switch"]))
col2.metric("Live trading", str(risk_state["live_trading"]))
col3.metric("Equity", f"{portfolio['equity']:.2f}")
col4.metric("Open positions", str(risk_state["open_positions"]))

st.subheader("Risk Limits")
st.dataframe(
    pd.DataFrame([{"limit": key, "value": value} for key, value in risk_limits.items()]),
    use_container_width=True,
)

st.subheader("Risk State")
st.json(
    {
        "available_margin": portfolio["available_margin"],
        "margin_used": portfolio["margin_used"],
        "order_count": risk_state["order_count"],
        "max_positions_reached": risk_state["max_positions_reached"],
    }
)
