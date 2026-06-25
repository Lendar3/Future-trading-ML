from __future__ import annotations

import pandas as pd


def risk_regime_filter(features: pd.DataFrame) -> pd.Series:
    risk_off = features.get("feature_risk_off_flag", pd.Series(0, index=features.index))
    return risk_off.fillna(0).astype(int)
