from __future__ import annotations

import pandas as pd


def build_liquidation_features(frame: pd.DataFrame) -> pd.DataFrame:
    long_liq = frame.get("liq_long_volume", pd.Series(0.0, index=frame.index)).astype(float)
    short_liq = frame.get("liq_short_volume", pd.Series(0.0, index=frame.index)).astype(float)
    out = pd.DataFrame(index=frame.index)
    out["feature_liq_long_volume_1m"] = long_liq
    out["feature_liq_short_volume_1m"] = short_liq
    out["feature_liq_long_count"] = frame.get("liq_long_count", 0)
    out["feature_liq_short_count"] = frame.get("liq_short_count", 0)
    out["feature_liq_imbalance"] = (short_liq - long_liq) / (short_liq + long_liq).replace(0, pd.NA)
    out["feature_liq_burst_30s"] = (
        (short_liq + long_liq) > (short_liq + long_liq).rolling(30).mean() * 3
    ).astype(int)
    out["feature_liq_cascade_flag"] = out["feature_liq_burst_30s"]
    out["feature_liq_near_price_flag"] = 0
    return out
