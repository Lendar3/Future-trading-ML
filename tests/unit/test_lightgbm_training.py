"""Unit tests for v1.7.2: Evaluation report + baseline comparison.

Coverage
--------
chronological_split
  * rows sorted by event_time, no shuffle
  * train/valid/test sizes respect requested fractions
  * no row appears in two splits
  * raises on impossible fractions

train_lightgbm_advisor
  * returns a TrainingResult with all required fields
  * model is fitted and can predict
  * all required metric keys present
  * chronological ordering preserved (train max_time < valid min_time < test min_time)
  * NaN features filled; NaN ratio recorded in metrics when non-zero
  * works with v1 dataset (OHLCV only)
  * works with enriched dataset (v1.6_public)
  * deterministic: same dataset → same model_run_id and same test_accuracy
  * raises ValueError when too few rows

_derive_target
  * 3-class derivation from label_long_edge + label_short_edge
  * prefers label_class when present

save_model_artifact / load_model_artifact
  * model.txt is written to disk
  * metadata.json contains expected keys
  * evaluation.json contains baseline/class-distribution report
  * load_model_artifact returns correct ModelArtifact
  * load_lgbm_model returns a usable lgb.Booster

live execution still disabled
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from app.core.safety import LiveTradingDisabledError
from app.execution.real_disabled import RealBrokerInterface
from app.features.dataset_builder import build_dataset_from_bars
from app.models.registry import (
    load_lgbm_model,
    load_model_artifact,
    load_model_evaluation,
    save_model_artifact,
)
from app.models.train_lightgbm import (
    TrainingResult,
    _derive_target,
    chronological_split,
    train_lightgbm_advisor,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE = datetime(2026, 1, 1, tzinfo=UTC)


def _make_bars(n: int = 120, start: datetime | None = None) -> pd.DataFrame:
    base = start or _BASE
    times = [base + timedelta(minutes=i) for i in range(n)]
    rng = np.random.default_rng(42)
    prices = 100.0 + np.cumsum(rng.normal(0, 0.1, n))
    return pd.DataFrame(
        {
            "event_time": times,
            "open": prices,
            "high": prices + 0.1,
            "low": prices - 0.1,
            "close": prices + rng.normal(0, 0.05, n),
            "volume": rng.uniform(5, 15, n),
        }
    )


def _make_dataset(n: int = 120) -> pd.DataFrame:
    bars = _make_bars(n)
    ds = build_dataset_from_bars(bars=bars, symbol="BTCUSDT", timeframe="1m", label_horizon=3)
    label_cols = [c for c in ds.columns if c.startswith("label_")]
    return ds.dropna(subset=label_cols).copy()


# ---------------------------------------------------------------------------
# chronological_split tests
# ---------------------------------------------------------------------------


class TestChronologicalSplit:
    def test_sizes_respect_fractions(self) -> None:
        df = _make_dataset(120)
        train, valid, test = chronological_split(df, train_frac=0.7, valid_frac=0.15)
        n = len(df)
        assert len(train) == int(n * 0.70)
        assert len(valid) == int(n * 0.70 + n * 0.15) - int(n * 0.70)
        assert len(test) == n - int(n * 0.70 + n * 0.15)
        assert len(train) + len(valid) + len(test) == n

    def test_sorted_by_event_time(self) -> None:
        """Each split must be chronologically ordered and non-overlapping."""
        df = _make_dataset(120)
        train, valid, test = chronological_split(df)
        assert train["event_time"].is_monotonic_increasing
        assert valid["event_time"].is_monotonic_increasing
        assert test["event_time"].is_monotonic_increasing

    def test_no_row_in_two_splits(self) -> None:
        df = _make_dataset(120)
        train, valid, test = chronological_split(df)
        all_idx = pd.concat([train, valid, test]).index
        assert len(all_idx) == len(set(all_idx)), "Duplicate rows across splits"

    def test_train_ends_before_valid_starts(self) -> None:
        df = _make_dataset(120)
        train, valid, test = chronological_split(df)
        assert train["event_time"].max() < valid["event_time"].min()
        assert valid["event_time"].max() < test["event_time"].min()

    def test_raises_on_impossible_fractions(self) -> None:
        df = _make_dataset(60)
        with pytest.raises(ValueError, match="must be < 1.0"):
            chronological_split(df, train_frac=0.8, valid_frac=0.3)

    def test_no_shuffle_across_calls(self) -> None:
        """Calling split twice on the same data must yield identical results."""
        df = _make_dataset(120)
        tr1, va1, te1 = chronological_split(df)
        tr2, va2, te2 = chronological_split(df)
        pd.testing.assert_frame_equal(tr1.reset_index(drop=True), tr2.reset_index(drop=True))


# ---------------------------------------------------------------------------
# train_lightgbm_advisor tests
# ---------------------------------------------------------------------------


class TestTrainLightgbmAdvisor:
    def test_returns_training_result(self) -> None:
        ds = _make_dataset(120)
        result = train_lightgbm_advisor(ds)
        assert isinstance(result, TrainingResult)

    def test_all_required_metric_keys_present(self) -> None:
        ds = _make_dataset(120)
        result = train_lightgbm_advisor(ds)
        required = {
            "rows_total",
            "rows_train",
            "rows_valid",
            "rows_test",
            "feature_count",
            "train_accuracy",
            "valid_accuracy",
            "test_accuracy",
            "valid_log_loss",
            "test_log_loss",
            "baseline_majority_accuracy",
            "baseline_random_accuracy",
            "model_vs_majority_accuracy_delta",
            "model_vs_random_accuracy_delta",
            "target_class_count",
            "target_high_imbalance_flag",
            "best_iteration",
            "early_stopping_enabled",
            "nan_mode",
            "class_weight_mode",
        }
        assert required.issubset(set(result.metrics.keys()))

    def test_model_is_fitted_and_can_predict(self) -> None:
        ds = _make_dataset(120)
        result = train_lightgbm_advisor(ds)
        assert result.model is not None
        X = ds[result.feature_columns].fillna(0.0).astype(np.float32)
        preds = result.model.predict(X)
        assert len(preds) == len(X)

    def test_accuracies_in_valid_range(self) -> None:
        ds = _make_dataset(120)
        result = train_lightgbm_advisor(ds)
        for key in ("train_accuracy", "valid_accuracy", "test_accuracy"):
            v = result.metrics[key]
            assert 0.0 <= v <= 1.0, f"{key}={v} out of range"

    def test_log_loss_positive(self) -> None:
        ds = _make_dataset(120)
        result = train_lightgbm_advisor(ds)
        assert result.metrics["valid_log_loss"] > 0
        assert result.metrics["test_log_loss"] > 0

    def test_rows_sum_to_total(self) -> None:
        ds = _make_dataset(120)
        result = train_lightgbm_advisor(ds)
        m = result.metrics
        assert int(m["rows_train"]) + int(m["rows_valid"]) + int(m["rows_test"]) == int(
            m["rows_total"]
        )

    def test_feature_columns_populated(self) -> None:
        ds = _make_dataset(120)
        result = train_lightgbm_advisor(ds)
        assert len(result.feature_columns) > 0
        assert all(c.startswith("feature_") for c in result.feature_columns)

    def test_label_column_populated(self) -> None:
        ds = _make_dataset(120)
        result = train_lightgbm_advisor(ds)
        assert result.label_column != ""

    def test_deterministic_same_data_same_run_id(self) -> None:
        ds = _make_dataset(120)
        r1 = train_lightgbm_advisor(ds)
        r2 = train_lightgbm_advisor(ds)
        assert r1.model_run_id == r2.model_run_id

    def test_deterministic_same_accuracy(self) -> None:
        ds = _make_dataset(120)
        r1 = train_lightgbm_advisor(ds)
        r2 = train_lightgbm_advisor(ds)
        assert r1.metrics["test_accuracy"] == r2.metrics["test_accuracy"]
        assert r1.metrics["baseline_majority_accuracy"] == r2.metrics["baseline_majority_accuracy"]
        assert r1.metrics["baseline_random_accuracy"] == r2.metrics["baseline_random_accuracy"]

    def test_class_distribution_metrics_exist(self) -> None:
        ds = _make_dataset(120)
        result = train_lightgbm_advisor(ds)
        for key in (
            "class_count_0",
            "class_count_1",
            "class_count_2",
            "class_ratio_0",
            "class_ratio_1",
            "class_ratio_2",
            "target_class_count",
            "target_high_imbalance_flag",
        ):
            assert key in result.metrics

    def test_raises_on_too_few_rows(self) -> None:
        ds = _make_dataset(20)
        with pytest.raises(ValueError, match="Too few rows"):
            train_lightgbm_advisor(ds)

    def test_nan_features_recorded_in_metrics(self) -> None:
        """NaN ratio must appear in metrics for a feature that is fully NaN."""
        ds = _make_dataset(120)
        # Corrupt one feature entirely.
        feat = ds.columns[ds.columns.str.startswith("feature_")][0]
        ds[feat] = float("nan")
        result = train_lightgbm_advisor(ds)
        assert any(k.startswith("nan_ratio_") for k in result.metrics)

    def test_no_lookahead_chronological_order(self) -> None:
        """Train partition must end before validation starts (time-order check)."""
        ds = _make_dataset(120)
        train_lightgbm_advisor(ds)
        # We cannot directly inspect split boundaries from TrainingResult,
        # but we can re-run chronological_split on the same data and verify.
        from app.models.train_lightgbm import chronological_split

        sorted_ds = ds.sort_values("event_time").reset_index(drop=True)
        train, valid, test = chronological_split(sorted_ds)
        assert train["event_time"].max() < valid["event_time"].min()
        assert valid["event_time"].max() < test["event_time"].min()

    def test_raises_on_no_feature_columns(self) -> None:
        bare = pd.DataFrame(
            {"event_time": [_BASE], "label_long_edge": [1], "label_short_edge": [0]}
        )
        with pytest.raises(ValueError, match="no feature_"):
            train_lightgbm_advisor(bare)

    def test_raises_on_no_label_columns(self) -> None:
        ds = _make_dataset(120)
        # Drop all label columns
        label_cols = [c for c in ds.columns if c.startswith("label_")]
        ds = ds.drop(columns=label_cols)
        with pytest.raises(ValueError, match="no label_"):
            train_lightgbm_advisor(ds)

    def test_early_stopping_metrics_exist(self) -> None:
        ds = _make_dataset(120)
        result = train_lightgbm_advisor(ds, early_stopping_rounds=20)
        assert result.metrics["early_stopping_enabled"] == 1.0
        assert result.metrics["best_iteration"] >= 1.0

    def test_default_class_weight_mode_is_none(self) -> None:
        ds = _make_dataset(120)
        result = train_lightgbm_advisor(ds)
        assert result.best_params["class_weight_mode"] == "none"
        assert result.metrics["class_weight_mode"] == 0.0

    def test_balanced_class_weight_mode_reflected(self) -> None:
        ds = _make_dataset(120)
        result = train_lightgbm_advisor(ds, class_weight_mode="balanced")
        assert result.best_params["class_weight_mode"] == "balanced"
        assert result.best_params["class_weight"] == "balanced"
        assert result.metrics["class_weight_mode"] == 1.0
        assert isinstance(result.best_params["lgbm_params"], dict)
        assert isinstance(result.best_params["training_config"], dict)
        assert result.best_params["training_config"]["class_weight_mode"] == "balanced"
        assert result.best_params["lgbm_params"]["class_weight"] == "balanced"

    def test_lgbm_constructor_does_not_receive_nan_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ds = _make_dataset(120)
        captured: dict[str, object] = {}

        class _StubEstimator:
            def __init__(self, **kwargs):  # noqa: ANN003
                captured.update(kwargs)
                self.n_estimators = int(kwargs.get("n_estimators", 100))
                self.best_iteration_ = self.n_estimators

            def fit(self, X, y, **kwargs):  # noqa: ANN001, ANN003
                self._classes = np.unique(y)
                return self

            def predict(self, X):  # noqa: ANN001
                return np.zeros(len(X), dtype=int)

            def predict_proba(self, X):  # noqa: ANN001
                n = len(X)
                class_count = max(len(getattr(self, "_classes", [0, 1])), 2)
                return np.full((n, class_count), 1.0 / class_count, dtype=float)

        monkeypatch.setattr("app.models.train_lightgbm.lgb.LGBMClassifier", _StubEstimator)
        train_lightgbm_advisor(ds, nan_mode="native")
        assert "nan_mode" not in captured

    def test_lgbm_constructor_does_not_receive_class_weight_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ds = _make_dataset(120)
        captured: dict[str, object] = {}

        class _StubEstimator:
            def __init__(self, **kwargs):  # noqa: ANN003
                captured.update(kwargs)
                self.n_estimators = int(kwargs.get("n_estimators", 100))
                self.best_iteration_ = self.n_estimators

            def fit(self, X, y, **kwargs):  # noqa: ANN001, ANN003
                self._classes = np.unique(y)
                return self

            def predict(self, X):  # noqa: ANN001
                return np.zeros(len(X), dtype=int)

            def predict_proba(self, X):  # noqa: ANN001
                n = len(X)
                class_count = max(len(getattr(self, "_classes", [0, 1])), 2)
                return np.full((n, class_count), 1.0 / class_count, dtype=float)

        monkeypatch.setattr("app.models.train_lightgbm.lgb.LGBMClassifier", _StubEstimator)
        train_lightgbm_advisor(ds, class_weight_mode="balanced")
        assert "class_weight_mode" not in captured
        assert captured.get("class_weight") == "balanced"

    def test_lgbm_constructor_does_not_receive_early_stopping_rounds(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ds = _make_dataset(120)
        captured: dict[str, object] = {}

        class _StubEstimator:
            def __init__(self, **kwargs):  # noqa: ANN003
                captured.update(kwargs)
                self.n_estimators = int(kwargs.get("n_estimators", 100))
                self.best_iteration_ = self.n_estimators

            def fit(self, X, y, **kwargs):  # noqa: ANN001, ANN003
                self._classes = np.unique(y)
                return self

            def predict(self, X):  # noqa: ANN001
                return np.zeros(len(X), dtype=int)

            def predict_proba(self, X):  # noqa: ANN001
                n = len(X)
                class_count = max(len(getattr(self, "_classes", [0, 1])), 2)
                return np.full((n, class_count), 1.0 / class_count, dtype=float)

        monkeypatch.setattr("app.models.train_lightgbm.lgb.LGBMClassifier", _StubEstimator)
        train_lightgbm_advisor(ds, early_stopping_rounds=10)
        assert "early_stopping_rounds" not in captured

    def test_nan_mode_metric_and_behavior_zero_fill(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ds = _make_dataset(120)
        feature = next(c for c in ds.columns if c.startswith("feature_"))
        ds.loc[ds.index[:10], feature] = np.nan

        original_fit = lgb.LGBMClassifier.fit
        seen = {"has_nan": False}

        def _fit(self, X, y, *args, **kwargs):  # noqa: ANN001
            seen["has_nan"] = bool(pd.DataFrame(X).isna().any().any())
            return original_fit(self, X, y, *args, **kwargs)

        monkeypatch.setattr("app.models.train_lightgbm.lgb.LGBMClassifier.fit", _fit)
        result = train_lightgbm_advisor(ds, nan_mode="zero_fill")
        assert result.metrics["nan_mode"] == 0.0
        assert seen["has_nan"] is False

    def test_nan_mode_metric_and_behavior_native(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ds = _make_dataset(120)
        feature = next(c for c in ds.columns if c.startswith("feature_"))
        ds.loc[ds.index[:10], feature] = np.nan

        original_fit = lgb.LGBMClassifier.fit
        seen = {"has_nan": False}

        def _fit(self, X, y, *args, **kwargs):  # noqa: ANN001
            seen["has_nan"] = bool(pd.DataFrame(X).isna().any().any())
            return original_fit(self, X, y, *args, **kwargs)

        monkeypatch.setattr("app.models.train_lightgbm.lgb.LGBMClassifier.fit", _fit)
        result = train_lightgbm_advisor(ds, nan_mode="native")
        assert result.metrics["nan_mode"] == 1.0
        assert seen["has_nan"] is True

    def test_derive_target_prefers_label_class(self) -> None:
        df = pd.DataFrame(
            {
                "label_class": [0, 1, 2],
                "label_long_edge": [1, 0, 1],
                "label_short_edge": [0, 1, 0],
            }
        )
        y, label = _derive_target(df, allow_binary_long_edge_fallback=True)
        assert label == "label_class"
        assert y.tolist() == [0, 1, 2]

    def test_derive_target_from_long_short_edges(self) -> None:
        df = pd.DataFrame({"label_long_edge": [0, 1, 0], "label_short_edge": [0, 0, 1]})
        y, label = _derive_target(df, allow_binary_long_edge_fallback=True)
        assert label == "label_long_edge+label_short_edge→3class"
        assert y.tolist() == [0, 1, 2]

    def test_derive_target_binary_fallback_when_explicitly_allowed(self) -> None:
        df = pd.DataFrame({"label_long_edge": [0, 1, 1]})
        y, label = _derive_target(df, allow_binary_long_edge_fallback=True)
        assert label == "label_long_edge"
        assert y.tolist() == [0, 1, 1]

    def test_derive_target_rejects_unsupported_label_columns(self) -> None:
        df = pd.DataFrame({"label_price_move": [1, 0, 1]})
        with pytest.raises(ValueError, match="Unsupported target labels"):
            _derive_target(df, allow_binary_long_edge_fallback=False)


# ---------------------------------------------------------------------------
# save / load artifact tests
# ---------------------------------------------------------------------------


class TestModelRegistry:
    def test_save_creates_model_txt_and_metadata(self) -> None:
        ds = _make_dataset(120)
        result = train_lightgbm_advisor(ds)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_model_artifact(result, base_dir=tmp)
            assert Path(path).exists(), "model.txt not created"
            assert (Path(tmp) / result.model_run_id / "metadata.json").exists()
            assert (Path(tmp) / result.model_run_id / "evaluation.json").exists()

    def test_load_model_artifact_returns_descriptor(self) -> None:
        ds = _make_dataset(120)
        result = train_lightgbm_advisor(ds)
        with tempfile.TemporaryDirectory() as tmp:
            save_model_artifact(result, base_dir=tmp)
            artifact = load_model_artifact(result.model_run_id, base_dir=tmp)
        assert artifact.model_run_id == result.model_run_id
        assert artifact.feature_columns == result.feature_columns
        assert artifact.label_column == result.label_column

    def test_metadata_has_expected_keys(self) -> None:
        import json

        ds = _make_dataset(120)
        result = train_lightgbm_advisor(ds)
        with tempfile.TemporaryDirectory() as tmp:
            save_model_artifact(result, base_dir=tmp)
            meta = json.loads((Path(tmp) / result.model_run_id / "metadata.json").read_text())
        expected_keys = {
            "model_run_id",
            "created_at",
            "artifact_path",
            "feature_columns",
            "label_column",
            "metrics",
            "params",
        }
        assert expected_keys.issubset(set(meta.keys()))

    def test_evaluation_json_is_loadable_and_has_expected_keys(self) -> None:
        ds = _make_dataset(120)
        result = train_lightgbm_advisor(ds)
        with tempfile.TemporaryDirectory() as tmp:
            save_model_artifact(result, base_dir=tmp)
            evaluation = load_model_evaluation(result.model_run_id, base_dir=tmp)
        required = {
            "model_run_id",
            "created_at",
            "label_column",
            "feature_count",
            "row_splits",
            "baseline_comparison",
            "class_distribution",
            "metrics",
        }
        assert required.issubset(set(evaluation.keys()))

    def test_load_lgbm_model_returns_booster(self) -> None:
        import numpy as np

        ds = _make_dataset(120)
        result = train_lightgbm_advisor(ds)
        with tempfile.TemporaryDirectory() as tmp:
            save_model_artifact(result, base_dir=tmp)
            booster = load_lgbm_model(result.model_run_id, base_dir=tmp)
        X = ds[result.feature_columns].fillna(0.0).to_numpy(dtype=np.float32)
        preds = booster.predict(X)
        assert preds.shape[0] == len(X)

    def test_load_raises_on_missing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, pytest.raises(FileNotFoundError):
            load_model_artifact("nonexistent_run_id", base_dir=tmp)

    def test_save_raises_when_no_model(self) -> None:
        result = TrainingResult(
            model_run_id="abc123",
            metrics={},
            best_params={},
            feature_columns=[],
            label_column="",
            model=None,
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            pytest.raises(ValueError, match="no fitted model"),
        ):
            save_model_artifact(result, base_dir=tmp)


# ---------------------------------------------------------------------------
# Live execution safety guard
# ---------------------------------------------------------------------------


def test_live_trading_still_disabled_v171() -> None:
    """Real broker must still raise LiveTradingDisabledError after v1.7.1."""
    broker = RealBrokerInterface()
    with pytest.raises(LiveTradingDisabledError):
        broker.place_order({"symbol": "BTCUSDT", "side": "Buy", "qty": 0.001})
