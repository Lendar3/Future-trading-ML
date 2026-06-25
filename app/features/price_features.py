from __future__ import annotations

import numpy as np
import pandas as pd


def build_price_features(bars: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=bars.index)
    close = bars["close"].astype(float)
    open_ = bars["open"].astype(float)
    out["feature_ret_1m"] = close.pct_change(1)
    out["feature_ret_5m"] = close.pct_change(5)
    out["feature_ret_15m"] = close.pct_change(15)
    out["feature_ret_1h"] = close.pct_change(60)
    price_ratio = close / close.shift(1)
    valid_ratio = price_ratio > 0.0
    log_return = pd.Series(np.nan, index=bars.index, dtype="float64")
    log_return.loc[valid_ratio] = np.log(price_ratio.loc[valid_ratio])
    out["feature_log_return"] = log_return
    out["feature_close_vs_open"] = close / open_ - 1.0
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    out["feature_close_vs_ema_20"] = close / ema20 - 1.0
    out["feature_close_vs_ema_50"] = close / ema50 - 1.0
    out["feature_zscore_close_20"] = (close - close.rolling(20).mean()) / close.rolling(20).std()
    out["feature_zscore_close_50"] = (close - close.rolling(50).mean()) / close.rolling(50).std()
    out["feature_distance_to_high_24h"] = close / close.rolling(1440, min_periods=1).max() - 1.0
    out["feature_distance_to_low_24h"] = close / close.rolling(1440, min_periods=1).min() - 1.0
    return out
