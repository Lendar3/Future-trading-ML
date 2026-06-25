from __future__ import annotations

import pandas as pd


def build_oi_features(frame: pd.DataFrame) -> pd.DataFrame:
    oi = frame["open_interest"].astype(float)
    close = frame.get("close", pd.Series(index=frame.index, dtype=float)).astype(float)
    volume = frame.get("volume", pd.Series(1.0, index=frame.index)).astype(float)
    out = pd.DataFrame(index=frame.index)
    out["feature_open_interest"] = oi
    out["feature_oi_delta_5m"] = oi.diff(5)
    out["feature_oi_delta_1h"] = oi.diff(60)
    out["feature_oi_zscore"] = (oi - oi.rolling(100).mean()) / oi.rolling(100).std()
    out["feature_oi_vs_volume"] = oi / volume.replace(0, pd.NA)
    price_delta = close.diff()
    oi_delta = oi.diff()
    out["feature_price_up_oi_up"] = ((price_delta > 0) & (oi_delta > 0)).astype(int)
    out["feature_price_up_oi_down"] = ((price_delta > 0) & (oi_delta < 0)).astype(int)
    out["feature_price_down_oi_up"] = ((price_delta < 0) & (oi_delta > 0)).astype(int)
    out["feature_price_down_oi_down"] = ((price_delta < 0) & (oi_delta < 0)).astype(int)
    out["feature_squeeze_risk"] = (out["feature_oi_zscore"] > 2).astype(int)
    return out
