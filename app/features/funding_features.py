from __future__ import annotations

import pandas as pd


def build_funding_features(funding: pd.DataFrame) -> pd.DataFrame:
    rate = funding["funding_rate"].astype(float)
    out = pd.DataFrame(index=funding.index)
    out["feature_funding_last"] = rate
    out["feature_funding_zscore_7d"] = (rate - rate.rolling(21).mean()) / rate.rolling(21).std()
    out["feature_funding_zscore_30d"] = (rate - rate.rolling(90).mean()) / rate.rolling(90).std()
    out["feature_expected_funding_cost"] = rate
    out["feature_funding_pressure_side"] = rate.apply(
        lambda value: 1 if value > 0 else -1 if value < 0 else 0
    )
    out["feature_funding_positive_streak"] = (
        (rate > 0).astype(int).groupby((rate <= 0).cumsum()).cumsum()
    )
    out["feature_funding_negative_streak"] = (
        (rate < 0).astype(int).groupby((rate >= 0).cumsum()).cumsum()
    )
    out["feature_minutes_to_next_funding"] = funding.get("minutes_to_next_funding", 0)
    out["feature_funding_next_estimated"] = funding.get("funding_next_estimated", rate)
    return out
