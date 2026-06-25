from __future__ import annotations

import pandas as pd

from app.features.dataset_builder import build_dataset_from_bars


def test_dataset_builder_adds_feature_and_label_prefixes() -> None:
    bars = pd.read_csv("tests/fixtures/sample_bars.csv", parse_dates=["event_time"])
    dataset = build_dataset_from_bars(bars, symbol="BTCUSDT", timeframe="1m", label_horizon=1)
    assert any(column.startswith("feature_") for column in dataset.columns)
    assert any(column.startswith("label_") for column in dataset.columns)
    assert (dataset["source_available_time"] <= dataset["event_time"]).all()
