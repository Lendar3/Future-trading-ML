from __future__ import annotations

import pandas as pd


def score_orderbook_imbalance(features: pd.DataFrame) -> pd.Series:
    return features.get("feature_obi_l1", pd.Series(0.0, index=features.index)).fillna(0.0)
