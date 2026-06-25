from __future__ import annotations

import pandas as pd


def build_volume_features(bars: pd.DataFrame) -> pd.DataFrame:
    volume = bars["volume"].astype(float)
    close = bars["close"].astype(float)
    out = pd.DataFrame(index=bars.index)
    out["feature_volume_1m"] = volume
    out["feature_volume_5m"] = volume.rolling(5).sum()
    out["feature_turnover_1m"] = volume * close
    out["feature_turnover_1h"] = out["feature_turnover_1m"].rolling(60).sum()
    out["feature_volume_vs_avg_20"] = volume / volume.rolling(20).mean()
    out["feature_volume_vs_avg_100"] = volume / volume.rolling(100).mean()
    out["feature_volume_zscore"] = (volume - volume.rolling(100).mean()) / volume.rolling(100).std()
    out["feature_volume_spike_flag"] = (out["feature_volume_zscore"] > 2.0).astype(int)
    return out
