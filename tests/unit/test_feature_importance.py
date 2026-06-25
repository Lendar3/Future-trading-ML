from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.core.safety import LiveTradingDisabledError
from app.execution.real_disabled import RealBrokerInterface
from app.features.dataset_builder import build_dataset_from_bars
from app.models.feature_importance import build_feature_importance_summary
from app.models.registry import save_model_artifact
from app.models.train_lightgbm import train_lightgbm_advisor


def _make_dataset(n: int = 160) -> pd.DataFrame:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    times = [base + timedelta(minutes=i) for i in range(n)]
    rng = np.random.default_rng(123)
    prices = 100.0 + np.cumsum(rng.normal(0, 0.12, n))
    bars = pd.DataFrame(
        {
            "event_time": times,
            "open": prices,
            "high": prices + 0.2,
            "low": prices - 0.2,
            "close": prices + rng.normal(0, 0.05, n),
            "volume": rng.uniform(5, 15, n),
        }
    )
    dataset = build_dataset_from_bars(bars=bars, symbol="BTCUSDT", timeframe="1m", label_horizon=3)
    label_cols = [c for c in dataset.columns if c.startswith("label_")]
    return dataset.dropna(subset=label_cols).copy()


def test_feature_importance_summary_supports_gain_and_top_n() -> None:
    ds = _make_dataset()
    trained = train_lightgbm_advisor(ds)
    with tempfile.TemporaryDirectory() as tmp:
        save_model_artifact(trained, base_dir=tmp)
        summary = build_feature_importance_summary(
            trained.model_run_id,
            importance_type="gain",
            top_n=7,
            base_dir=tmp,
        )
    assert summary["model_run_id"] == trained.model_run_id
    assert summary["importance_type"] == "gain"
    assert len(summary["top_features"]) <= 7
    assert int(summary["feature_count"]) == len(summary["all_features"])


def test_feature_importance_summary_supports_split() -> None:
    ds = _make_dataset()
    trained = train_lightgbm_advisor(ds)
    with tempfile.TemporaryDirectory() as tmp:
        save_model_artifact(trained, base_dir=tmp)
        summary = build_feature_importance_summary(
            trained.model_run_id,
            importance_type="split",
            top_n=10,
            base_dir=tmp,
        )
    assert summary["importance_type"] == "split"
    assert len(summary["top_features"]) <= 10


def test_importance_share_sums_to_one_when_total_positive() -> None:
    ds = _make_dataset()
    trained = train_lightgbm_advisor(ds)
    with tempfile.TemporaryDirectory() as tmp:
        save_model_artifact(trained, base_dir=tmp)
        summary = build_feature_importance_summary(
            trained.model_run_id,
            importance_type="gain",
            top_n=20,
            base_dir=tmp,
        )
    rows = list(summary["all_features"])
    total_importance = sum(float(row["importance"]) for row in rows)
    share_sum = sum(float(row["importance_share"]) for row in rows)
    if total_importance > 0:
        assert share_sum == pytest.approx(1.0, abs=1e-6)
    else:
        assert share_sum == pytest.approx(0.0, abs=1e-12)


def test_unsupported_importance_type_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported importance_type"):
        build_feature_importance_summary("missing", importance_type="bad_type")


def test_live_execution_remains_disabled() -> None:
    broker = RealBrokerInterface()
    with pytest.raises(LiveTradingDisabledError):
        broker.place_order({"symbol": "BTCUSDT", "side": "Buy", "qty": 0.001})
