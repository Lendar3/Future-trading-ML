from __future__ import annotations

import pandas as pd
import streamlit as st


def render_table(frame: pd.DataFrame) -> None:
    st.dataframe(frame, use_container_width=True)
