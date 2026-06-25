from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.core.safety import LiveTradingDisabledError
from app.execution.real_disabled import RealBrokerInterface
from app.features.dataset_builder import build_dataset_from_bars
from app.models.predict import predict_advisor_probabilities
from app.models.registry import save_model_artifact
from app.models.train_lightgbm import train_lightgbm_advisor
from app.strategies.advisor_ml import predictions_to_signal_frame


def _bars(n: int = 120) -> pd.DataFrame:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    times = [base + timedelta(minutes=i) for i in range(n)]
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 0.08, n))
    return pd.DataFrame(
        {
            "event_time": times,
            "open": close,
            "high": close + 0.12,
            "low": close - 0.12,
            "close": close + rng.normal(0, 0.04, n),
            "volume": rng.uniform(5, 15, n),
        }
    )


def test_prediction_uses_artifact_feature_order_and_fills_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = pd.DataFrame(
        {
            "f1": [1.0, np.nan],
            "f2": [2.0, 3.0],
            "extra": [9.0, 9.0],
        },
        index=[10, 11],
    )

    captured: dict[str, pd.DataFrame] = {}

    class DummyBooster:
        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            captured["frame"] = frame.copy()
            return np.array([[0.2, 0.7, 0.1], [0.4, 0.3, 0.3]], dtype=np.float32)

    monkeypatch.setattr(
        "app.models.predict.load_model_artifact",
        lambda model_run_id, base_dir: SimpleNamespace(feature_columns=["f2", "f_missing", "f1"]),
    )
    monkeypatch.setattr(
        "app.models.predict.load_lgbm_model",
        lambda model_run_id, base_dir: DummyBooster(),
    )

    out = predict_advisor_probabilities(features=features, model_run_id="m1")
    used = captured["frame"]
    assert list(used.columns) == ["f2", "f_missing", "f1"]
    assert float(used.loc[10, "f_missing"]) == 0.0
    assert float(used.loc[11, "f1"]) == 0.0
    assert "extra" not in used.columns
    assert out.index.tolist() == [10, 11]


def test_real_saved_artifact_produces_probability_columns(tmp_path: Path) -> None:
    ds = build_dataset_from_bars(_bars(140), symbol="BTCUSDT", timeframe="1m")
    label_cols = [c for c in ds.columns if c.startswith("label_")]
    ds = ds.dropna(subset=label_cols).copy()
    result = train_lightgbm_advisor(ds)
    save_model_artifact(result, base_dir=tmp_path)

    out = predict_advisor_probabilities(
        features=ds,
        model_run_id=result.model_run_id,
        artifact_base_dir=tmp_path,
    )
    assert "feature_ml_no_edge_probability" in out.columns
    assert "feature_ml_long_probability" in out.columns
    assert "feature_ml_short_probability" in out.columns
    assert len(out) == len(ds)


def test_unsupported_prediction_shape_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class BadBooster:
        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            return np.zeros((len(frame), 4), dtype=np.float32)

    monkeypatch.setattr(
        "app.models.predict.load_model_artifact",
        lambda model_run_id, base_dir: SimpleNamespace(feature_columns=["f1"]),
    )
    monkeypatch.setattr(
        "app.models.predict.load_lgbm_model",
        lambda model_run_id, base_dir: BadBooster(),
    )

    with pytest.raises(ValueError, match="Unsupported prediction shape"):
        predict_advisor_probabilities(
            features=pd.DataFrame({"f1": [1.0, 2.0]}),
            model_run_id="m1",
        )


def test_predictions_to_signal_frame_creates_long_signal() -> None:
    dataset = pd.DataFrame(
        [
            {
                "event_time": datetime(2026, 1, 1, tzinfo=UTC),
                "symbol": "BTCUSDT",
                "close": 100.0,
            }
        ]
    )
    probs = pd.DataFrame(
        [
            {
                "feature_ml_no_edge_probability": 0.20,
                "feature_ml_long_probability": 0.75,
                "feature_ml_short_probability": 0.05,
            }
        ],
        index=dataset.index,
    )
    sig = predictions_to_signal_frame(
        dataset, probs, model_run_id="run-x", symbol_fallback="BTCUSDT"
    )
    assert sig.iloc[0]["side"] == "long"


def test_predictions_to_signal_frame_creates_short_signal() -> None:
    dataset = pd.DataFrame(
        [
            {
                "event_time": datetime(2026, 1, 1, tzinfo=UTC),
                "symbol": "BTCUSDT",
                "close": 100.0,
            }
        ]
    )
    probs = pd.DataFrame(
        [
            {
                "feature_ml_no_edge_probability": 0.20,
                "feature_ml_long_probability": 0.10,
                "feature_ml_short_probability": 0.70,
            }
        ],
        index=dataset.index,
    )
    sig = predictions_to_signal_frame(
        dataset, probs, model_run_id="run-x", symbol_fallback="BTCUSDT"
    )
    assert sig.iloc[0]["side"] == "short"


def test_predictions_to_signal_frame_weak_probabilities_create_no_trade() -> None:
    dataset = pd.DataFrame(
        [
            {
                "event_time": datetime(2026, 1, 1, tzinfo=UTC),
                "symbol": "BTCUSDT",
                "close": 100.0,
            }
        ]
    )
    probs = pd.DataFrame(
        [
            {
                "feature_ml_no_edge_probability": 0.49,
                "feature_ml_long_probability": 0.51,
                "feature_ml_short_probability": 0.00,
            }
        ],
        index=dataset.index,
    )
    sig = predictions_to_signal_frame(
        dataset, probs, model_run_id="run-x", symbol_fallback="BTCUSDT"
    )
    assert sig.iloc[0]["side"] == "no_trade"


def test_rationale_json_contains_model_and_probabilities() -> None:
    dataset = pd.DataFrame(
        [
            {
                "event_time": datetime(2026, 1, 1, tzinfo=UTC),
                "symbol": "BTCUSDT",
                "close": 100.0,
            }
        ]
    )
    probs = pd.DataFrame(
        [
            {
                "feature_ml_no_edge_probability": 0.2,
                "feature_ml_long_probability": 0.6,
                "feature_ml_short_probability": 0.2,
            }
        ],
        index=dataset.index,
    )
    sig = predictions_to_signal_frame(
        dataset, probs, model_run_id="run-y", symbol_fallback="BTCUSDT"
    )
    rationale = sig.iloc[0]["rationale_json"]
    assert rationale["model_run_id"] == "run-y"
    assert "probabilities" in rationale


def test_live_execution_remains_disabled() -> None:
    broker = RealBrokerInterface()
    with pytest.raises(LiveTradingDisabledError):
        broker.place_order({"symbol": "BTCUSDT", "side": "Buy", "qty": 0.001})
