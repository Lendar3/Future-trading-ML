from __future__ import annotations

import pandas as pd


def build_long_short_features(frame: pd.DataFrame) -> pd.DataFrame:
    ratio = frame["long_short_ratio"].astype(float)
    out = pd.DataFrame(index=frame.index)
    out["feature_long_short_ratio"] = ratio
    out["feature_long_account_ratio"] = ratio / (1 + ratio)
    out["feature_short_account_ratio"] = 1 / (1 + ratio)
    out["feature_ls_ratio_zscore"] = (ratio - ratio.rolling(100).mean()) / ratio.rolling(100).std()
    out["feature_crowded_long_flag"] = (out["feature_ls_ratio_zscore"] > 2).astype(int)
    out["feature_crowded_short_flag"] = (out["feature_ls_ratio_zscore"] < -2).astype(int)
    out["feature_ls_ratio_change_1h"] = ratio.diff(60)
    return out
