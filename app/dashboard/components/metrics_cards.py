from __future__ import annotations

import streamlit as st


def render_metric(label: str, value: str | float) -> None:
    st.metric(label, value)
