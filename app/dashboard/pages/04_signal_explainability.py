from __future__ import annotations

import pandas as pd
import streamlit as st

from app.dashboard.api_client import DashboardApiError, get_json

st.title("Signal Explainability")

try:
    signals = get_json("/signals/latest", {"limit": 25})["signals"]
except DashboardApiError as exc:
    st.warning(str(exc))
    st.stop()

if not signals:
    st.info("No persisted signals yet. Run `make seed-demo` first.")
    st.stop()

signal_labels = [f"{item['symbol']} | {item['side']} | {item['signal_id']}" for item in signals]
selected_label = st.selectbox("Signal", signal_labels)
selected_signal_id = selected_label.split(" | ")[-1]

try:
    explanation = get_json(f"/signals/{selected_signal_id}/explain")
except DashboardApiError as exc:
    st.warning(str(exc))
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Decision", str(explanation["decision"]))
col2.metric("Strategy", str(explanation["strategy_name"]))
col3.metric("Mode", str(explanation["system_mode"]))

st.subheader("Rationale")
st.json(explanation["rationale"])

st.subheader("Risk Checks")
st.json(explanation["risk_checks"])

st.subheader("Cost Assumptions")
st.json(explanation["cost_assumptions"])

st.subheader("Feature Snapshot")
features = pd.DataFrame(
    [{"feature": key, "value": value} for key, value in explanation["feature_snapshot"].items()]
)
st.dataframe(features, use_container_width=True)
