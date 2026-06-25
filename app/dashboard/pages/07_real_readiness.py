from __future__ import annotations

import streamlit as st

from app.dashboard.api_client import DashboardApiError, get_json

st.title("Real Readiness")

try:
    checks = get_json("/ui/readiness-report")
except DashboardApiError as exc:
    st.warning(str(exc))
    st.stop()

for name, ok in checks.items():
    st.checkbox(name.replace("_", " "), value=bool(ok), disabled=True)
