from __future__ import annotations

import math

import pandas as pd
import pytest

from app.features.price_features import build_price_features


def test_log_return_handles_first_positive_and_invalid_rows() -> None:
    bars = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 110.0, 0.0, -5.0, 120.0],
            "volume": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    features = build_price_features(bars)
    series = features["feature_log_return"]

    assert pd.isna(series.iloc[0])
    assert series.iloc[1] == pytest.approx(math.log(1.1))
    assert pd.isna(series.iloc[2])
    assert pd.isna(series.iloc[3])
    assert pd.isna(series.iloc[4])
