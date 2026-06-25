from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from app.strategies.mean_reversion import generate_mean_reversion_signals


def test_mean_reversion_long_signal_emits_valid_refs() -> None:
    features = pd.DataFrame(
        {
            "event_time": [datetime(2026, 1, 1, tzinfo=UTC)],
            "close": [100.0],
            "feature_zscore_close_20": [-2.5],
            "feature_close_vs_ema_20": [0.01],
            "feature_atr_14": [2.0],
        }
    )

    signals = generate_mean_reversion_signals(features, symbol="BTCUSDT")
    row = signals.iloc[0]

    assert row["side"] == "long"
    assert row["stop_ref"] > 0.0
    assert row["take_ref"] > 0.0
    assert row["stop_ref"] < row["entry_ref"] < row["take_ref"]


def test_mean_reversion_short_signal_emits_valid_refs() -> None:
    features = pd.DataFrame(
        {
            "event_time": [datetime(2026, 1, 1, tzinfo=UTC)],
            "close": [100.0],
            "feature_zscore_close_20": [2.5],
            "feature_close_vs_ema_20": [-0.01],
            "feature_atr_14": [2.0],
        }
    )

    signals = generate_mean_reversion_signals(features, symbol="BTCUSDT")
    row = signals.iloc[0]

    assert row["side"] == "short"
    assert row["stop_ref"] > 0.0
    assert row["take_ref"] > 0.0
    assert row["take_ref"] < row["entry_ref"] < row["stop_ref"]
