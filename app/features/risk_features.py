from __future__ import annotations

import pandas as pd


def build_risk_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for name in (
        "current_equity",
        "margin_used",
        "available_margin",
        "position_size",
        "position_leverage",
        "daily_pnl",
        "daily_drawdown",
        "max_drawdown",
        "open_risk",
        "risk_per_trade",
        "kill_switch_active",
        "max_positions_reached",
    ):
        out[f"feature_{name}"] = frame.get(name, 0.0)
    return out
