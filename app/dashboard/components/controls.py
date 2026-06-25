from __future__ import annotations

import streamlit as st


def symbol_selector() -> str:
    value = st.selectbox("Symbol", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    return str(value)
