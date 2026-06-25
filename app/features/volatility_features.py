from __future__ import annotations

import pandas as pd


def build_volatility_features(bars: pd.DataFrame) -> pd.DataFrame:
    close = bars["close"].astype(float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    true_range = pd.concat(
        [(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1
    ).max(axis=1)
    out = pd.DataFrame(index=bars.index)
    returns = close.pct_change()
    out["feature_realized_vol_15m"] = returns.rolling(15).std()
    out["feature_realized_vol_1h"] = returns.rolling(60).std()
    out["feature_realized_vol_4h"] = returns.rolling(240).std()
    out["feature_true_range"] = true_range
    out["feature_atr_14"] = true_range.rolling(14).mean()
    out["feature_high_low_range_bp"] = (high / low - 1.0) * 10_000
    out["feature_intrabar_volatility"] = (high - low) / close
    out["feature_volatility_zscore"] = (
        out["feature_realized_vol_1h"] - out["feature_realized_vol_1h"].rolling(100).mean()
    ) / out["feature_realized_vol_1h"].rolling(100).std()
    out["feature_volatility_regime"] = (out["feature_volatility_zscore"] > 1.0).astype(int)
    return out
