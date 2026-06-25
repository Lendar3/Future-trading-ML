from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def candlestick_figure(bars: pd.DataFrame) -> go.Figure:
    figure = go.Figure(
        data=[
            go.Candlestick(
                x=bars["event_time"],
                open=bars["open"],
                high=bars["high"],
                low=bars["low"],
                close=bars["close"],
            )
        ]
    )
    figure.update_layout(
        height=420,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        xaxis_rangeslider_visible=False,
    )
    return figure


def volume_figure(bars: pd.DataFrame) -> go.Figure:
    figure = go.Figure(data=[go.Bar(x=bars["event_time"], y=bars["volume"])])
    figure.update_layout(
        height=220,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        yaxis_title="Volume",
    )
    return figure
