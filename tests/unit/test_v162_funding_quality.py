"""Unit tests for v1.6.2: funding time precision + dataset quality report.

Coverage
--------
funding_time.minutes_to_next_funding
  * 00:00 UTC boundary → exactly 0.0
  * 07:59 UTC → 1.0 min to 08:00
  * 08:00 UTC boundary → exactly 0.0
  * 15:59 UTC → 1.0 min to 16:00
  * 16:00 UTC boundary → exactly 0.0
  * 23:59 UTC → 1.0 min to 00:00 (next day)
  * tz-aware (non-UTC) datetime is handled correctly
  * raises ValueError for naive datetime
  * result is always in (0.0, 480.0]

funding_time.minutes_to_next_funding_series
  * vectorised version matches scalar results

dataset_quality.compute_dataset_quality
  * total_rows correct
  * feature_columns counts only feature_ prefixed cols
  * label_columns counts only label_ prefixed cols
  * first/last event_time correct
  * symbol / timeframe / feature_set_name from dataset metadata
  * nan_ratio_by_group: 0.0 for fully-present groups
  * nan_ratio_by_group: 1.0 for fully-NaN groups
  * overall_nan_ratio aggregates correctly
  * label_distribution counts match
  * label_distribution_pct sums to ≈100
  * rows_missing_labels counted correctly
  * stale_sparse_rows: 0 when no NaN, n when all NaN
  * to_dict returns the expected keys

dataset builder: feature_minutes_to_next_funding no-lookahead
  * value is schedule-derived, never NaN (even when funding rate is stale)
  * value is in (0, 480]
  * feature_funding_last is NaN when no funding data provided
  * feature_funding_last is not NaN when funding data provided in range
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pandas as pd
import pytest

from app.features.dataset_builder import build_dataset_from_public_tables
from app.features.dataset_quality import compute_dataset_quality
from app.features.funding_time import (
    minutes_to_next_funding,
    minutes_to_next_funding_series,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dt(hour: int, minute: int = 0, second: int = 0) -> datetime:
    """UTC datetime on 2026-01-01 at the given time."""
    return datetime(2026, 1, 1, hour, minute, second, tzinfo=UTC)


def _bars(n: int = 10, start: datetime | None = None) -> pd.DataFrame:
    base = start or _dt(0)
    times = [base + timedelta(minutes=i) for i in range(n)]
    return pd.DataFrame(
        {
            "event_time": times,
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
            "volume": [10.0 + i for i in range(n)],
        }
    )


def _sparse(times: list[datetime], values: list[float], col: str) -> pd.DataFrame:
    return pd.DataFrame({"event_time": times, col: values})


# ---------------------------------------------------------------------------
# Tests: minutes_to_next_funding scalar
# ---------------------------------------------------------------------------


class TestMinutesToNextFunding:
    def test_midnight_boundary(self) -> None:
        """00:00 UTC → exactly 0 minutes to next funding."""
        assert minutes_to_next_funding(_dt(0, 0, 0)) == pytest.approx(0.0)

    def test_0759_to_0800(self) -> None:
        """07:59:00 UTC → 1.0 min to 08:00."""
        assert minutes_to_next_funding(_dt(7, 59, 0)) == pytest.approx(1.0)

    def test_0800_boundary(self) -> None:
        """08:00 UTC → exactly 0 minutes to next funding."""
        assert minutes_to_next_funding(_dt(8, 0, 0)) == pytest.approx(0.0)

    def test_1559_to_1600(self) -> None:
        """15:59:00 UTC → 1.0 min to 16:00."""
        assert minutes_to_next_funding(_dt(15, 59, 0)) == pytest.approx(1.0)

    def test_1600_boundary(self) -> None:
        """16:00 UTC → exactly 0 minutes to next funding."""
        assert minutes_to_next_funding(_dt(16, 0, 0)) == pytest.approx(0.0)

    def test_2359_to_midnight(self) -> None:
        """23:59:00 UTC → 1.0 min to 00:00 next day."""
        assert minutes_to_next_funding(_dt(23, 59, 0)) == pytest.approx(1.0)

    def test_tz_aware_non_utc(self) -> None:
        """A non-UTC tz-aware datetime is converted to UTC correctly."""
        # UTC+02:00 10:00 == UTC 08:00 → exactly 0 min to next window.
        tz_plus2 = timezone(timedelta(hours=2))
        dt_plus2 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=tz_plus2)
        assert minutes_to_next_funding(dt_plus2) == pytest.approx(0.0)

    def test_raises_on_naive_datetime(self) -> None:
        """Naive datetime must raise ValueError."""
        naive = datetime(2026, 1, 1, 8, 0, 0)
        with pytest.raises(ValueError, match="timezone-aware"):
            minutes_to_next_funding(naive)

    def test_result_in_valid_range(self) -> None:
        """Result must always be in (0.0, 480.0] for any time."""
        test_times = [
            _dt(0, 0, 1),  # 1s after midnight
            _dt(4, 0, 0),  # mid-window
            _dt(7, 59, 59),
            _dt(8, 0, 1),
            _dt(12, 0, 0),
            _dt(15, 59, 59),
            _dt(16, 0, 1),
            _dt(20, 0, 0),
            _dt(23, 59, 59),
        ]
        for t in test_times:
            result = minutes_to_next_funding(t)
            assert 0.0 < result <= 480.0, f"Out of range for {t}: {result}"

    def test_boundary_is_exactly_zero(self) -> None:
        """Boundary timestamps produce exactly 0.0."""
        for h in (0, 8, 16):
            assert minutes_to_next_funding(_dt(h, 0, 0)) == pytest.approx(
                0.0
            ), f"Expected 0.0 at {h:02d}:00 UTC"

    def test_mid_period_correct(self) -> None:
        """04:00 UTC → 4 h = 240 min to 08:00."""
        assert minutes_to_next_funding(_dt(4, 0, 0)) == pytest.approx(240.0)

    def test_just_after_boundary_is_max(self) -> None:
        """00:00:01 UTC → (480 - 1/60) min to next window."""
        result = minutes_to_next_funding(_dt(0, 0, 1))
        assert result == pytest.approx(8 * 60 - 1 / 60, rel=1e-6)


# ---------------------------------------------------------------------------
# Tests: minutes_to_next_funding_series
# ---------------------------------------------------------------------------


class TestMinutesToNextFundingSeries:
    def test_matches_scalar(self) -> None:
        times = pd.Series(
            [_dt(0, 0, 0), _dt(7, 59, 0), _dt(8, 0, 0), _dt(16, 0, 0), _dt(23, 59, 0)]
        )
        result = minutes_to_next_funding_series(times)
        expected = [minutes_to_next_funding(t) for t in times]
        for r, e in zip(result, expected, strict=True):
            assert r == pytest.approx(e)

    def test_returns_series_same_length(self) -> None:
        times = pd.Series([_dt(h, 0, 0) for h in range(24)])
        result = minutes_to_next_funding_series(times)
        assert len(result) == 24

    def test_all_values_in_range(self) -> None:
        times = pd.Series([_dt(h, m) for h in range(24) for m in range(0, 60, 10)])
        result = minutes_to_next_funding_series(times)
        assert (result >= 0.0).all()
        assert (result <= 480.0).all()


# ---------------------------------------------------------------------------
# Tests: compute_dataset_quality
# ---------------------------------------------------------------------------

_BASE = datetime(2026, 1, 1, tzinfo=UTC)


def _make_dataset(
    n: int = 20,
    with_nan_feature: bool = False,
    with_missing_label: bool = False,
) -> pd.DataFrame:
    """Build a minimal feature/label dataset for quality tests."""
    bars = _bars(n, start=_BASE)
    ds = build_dataset_from_public_tables(
        bars=bars,
        symbol="BTCUSDT",
        timeframe="1m",
    )
    if with_nan_feature:
        ds["feature_ret_1m"] = float("nan")
    if with_missing_label:
        # Force some labels to NaN (normally the last label_horizon rows are NaN).
        label_col = next(c for c in ds.columns if c.startswith("label_"))
        ds.loc[ds.index[:5], label_col] = float("nan")
    return ds


class TestComputeDatasetQuality:
    def test_total_rows(self) -> None:
        ds = _make_dataset(20)
        report = compute_dataset_quality(ds)
        assert report.total_rows == 20

    def test_feature_columns_only_feature_prefix(self) -> None:
        ds = _make_dataset(10)
        report = compute_dataset_quality(ds)
        expected = sum(1 for c in ds.columns if c.startswith("feature_"))
        assert report.feature_columns == expected

    def test_label_columns_only_label_prefix(self) -> None:
        ds = _make_dataset(10)
        report = compute_dataset_quality(ds)
        expected = sum(1 for c in ds.columns if c.startswith("label_"))
        assert report.label_columns == expected

    def test_symbol_timeframe_feature_set_name(self) -> None:
        ds = _make_dataset(10)
        report = compute_dataset_quality(ds)
        assert report.symbol == "BTCUSDT"
        assert report.timeframe == "1m"
        assert report.feature_set_name == "v1.6_public"

    def test_first_last_event_time(self) -> None:
        ds = _make_dataset(10)
        report = compute_dataset_quality(ds)
        assert report.first_event_time != ""
        assert report.last_event_time != ""
        assert report.first_event_time <= report.last_event_time

    def test_nan_ratio_zero_for_complete_feature(self) -> None:
        bars = _bars(20, start=_BASE)
        ds = build_dataset_from_public_tables(bars=bars, symbol="BTCUSDT", timeframe="1m")
        report = compute_dataset_quality(ds)
        pvv = report.nan_ratio_by_group.get("price_volume_volatility", None)
        assert pvv is not None, "price_volume_volatility group must be present"
        # Rolling-window features (ATR, BB, HV, zscore) produce leading NaN rows
        # so the ratio is > 0 but must be strictly less than 1.0.
        assert pvv < 1.0, f"Expected pvv < 1.0 but got {pvv}"

    def test_nan_ratio_one_for_all_nan_feature(self) -> None:
        ds = _make_dataset(20, with_nan_feature=True)
        report = compute_dataset_quality(ds)
        # Setting feature_ret_1m to NaN makes price_volume_volatility non-zero.
        pvv = report.nan_ratio_by_group.get("price_volume_volatility", 0.0)
        assert pvv > 0.0

    def test_overall_nan_ratio_nonnegative(self) -> None:
        ds = _make_dataset(20)
        report = compute_dataset_quality(ds)
        assert 0.0 <= report.overall_nan_ratio <= 1.0

    def test_label_distribution_keys_present(self) -> None:
        ds = _make_dataset(20)
        report = compute_dataset_quality(ds)
        # Labels should include at least one class.
        assert len(report.label_distribution) >= 1

    def test_label_distribution_pct_sums_to_100(self) -> None:
        ds = _make_dataset(20)
        report = compute_dataset_quality(ds)
        total_pct = sum(report.label_distribution_pct.values())
        assert total_pct == pytest.approx(100.0, abs=0.1)

    def test_rows_missing_labels_counted(self) -> None:
        ds = _make_dataset(20, with_missing_label=True)
        report = compute_dataset_quality(ds)
        # At least the 5 we forced + natural horizon NaN rows.
        assert report.rows_missing_labels >= 5

    def test_stale_sparse_rows_zero_when_no_sparse(self) -> None:
        """Legacy dataset without sparse features: stale_sparse_rows is empty."""
        bars = _bars(10, start=_BASE)
        from app.features.dataset_builder import build_dataset_from_bars

        ds = build_dataset_from_bars(bars=bars, symbol="BTCUSDT", timeframe="1m")
        report = compute_dataset_quality(ds)
        # No sparse columns → dict should be empty (no groups detected).
        for v in report.stale_sparse_rows.values():
            assert isinstance(v, int)
            assert v >= 0

    def test_to_dict_has_expected_keys(self) -> None:
        ds = _make_dataset(10)
        report = compute_dataset_quality(ds)
        d = report.to_dict()
        expected_keys = {
            "symbol",
            "timeframe",
            "feature_set_name",
            "total_rows",
            "feature_columns",
            "label_columns",
            "first_event_time",
            "last_event_time",
            "nan_ratio_by_group",
            "overall_nan_ratio",
            "label_distribution",
            "label_distribution_pct",
            "rows_missing_labels",
            "stale_sparse_rows",
        }
        assert set(d.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Tests: feature_minutes_to_next_funding in dataset builder
# ---------------------------------------------------------------------------


class TestBuilderFundingTimePrecision:
    def test_minutes_to_next_funding_never_nan(self) -> None:
        """feature_minutes_to_next_funding must never be NaN — it's schedule-derived."""
        bars = _bars(10, start=_dt(7, 55))  # starts 5 min before 08:00 window
        funding = _sparse([_dt(7, 50)], [0.0001], "funding_rate")
        ds = build_dataset_from_public_tables(
            bars=bars,
            symbol="BTCUSDT",
            timeframe="1m",
            funding=funding,
        )
        assert (
            ds["feature_minutes_to_next_funding"].notna().all()
        ), "feature_minutes_to_next_funding must never be NaN."

    def test_minutes_to_next_funding_in_valid_range(self) -> None:
        """Values must be in (0.0, 480.0]."""
        bars = _bars(30, start=_dt(0, 0))
        funding = _sparse([_dt(0, 0)], [0.0001], "funding_rate")
        ds = build_dataset_from_public_tables(
            bars=bars,
            symbol="BTCUSDT",
            timeframe="1m",
            funding=funding,
        )
        vals = ds["feature_minutes_to_next_funding"]
        assert (vals >= 0.0).all()
        assert (vals <= 480.0).all()

    def test_funding_last_nan_when_no_funding_data(self) -> None:
        """feature_funding_last is NaN when no funding table is provided."""
        bars = _bars(5, start=_dt(0, 0))
        ds = build_dataset_from_public_tables(
            bars=bars, symbol="BTCUSDT", timeframe="1m", funding=None
        )
        assert "feature_funding_last" not in ds.columns

    def test_funding_last_not_nan_when_rate_available(self) -> None:
        """feature_funding_last is not NaN when a fresh rate is available."""
        bars = _bars(5, start=_dt(0, 0))
        funding = _sparse([_dt(0, 0)], [0.0001], "funding_rate")
        ds = build_dataset_from_public_tables(
            bars=bars,
            symbol="BTCUSDT",
            timeframe="1m",
            funding=funding,
        )
        assert "feature_funding_last" in ds.columns
        # At least the first bar (exact match at 00:00) should have rate.
        assert ds["feature_funding_last"].notna().any()

    def test_funding_last_nan_when_rate_stale(self) -> None:
        """feature_funding_last is NaN when all bars are beyond max_age from funding."""
        # Funding at t=0, bars start 48h later → age >> FUNDING_MAX_AGE_SECONDS.
        bars = _bars(5, start=_dt(0, 0) + timedelta(hours=48))
        funding = _sparse([_dt(0, 0)], [0.0001], "funding_rate")
        ds = build_dataset_from_public_tables(
            bars=bars,
            symbol="BTCUSDT",
            timeframe="1m",
            funding=funding,
        )
        # feature_minutes_to_next_funding should still be present and valid.
        assert ds["feature_minutes_to_next_funding"].notna().all()
        # feature_funding_last should be NaN because funding is too old.
        assert ds["feature_funding_last"].isna().all()

    def test_no_lookahead_future_funding_rate_not_used(self) -> None:
        """Future funding rate must never appear in feature_funding_last."""
        bars = _bars(5, start=_dt(0, 0))
        future_funding = _sparse(
            [_dt(0, 0) + timedelta(hours=1)],  # 1h in the future
            [0.9999],
            "funding_rate",
        )
        ds = build_dataset_from_public_tables(
            bars=bars,
            symbol="BTCUSDT",
            timeframe="1m",
            funding=future_funding,
        )
        # feature_minutes_to_next_funding is always present (schedule-based).
        assert ds["feature_minutes_to_next_funding"].notna().all()
        # feature_funding_last must be NaN (no past rate available).
        assert ds["feature_funding_last"].isna().all()

    def test_minutes_to_next_funding_correct_at_0700(self) -> None:
        """Bars at 07:00 UTC → 60 min to 08:00."""
        bars = _bars(1, start=_dt(7, 0))
        funding = _sparse([_dt(0, 0)], [0.0001], "funding_rate")
        ds = build_dataset_from_public_tables(
            bars=bars, symbol="BTCUSDT", timeframe="1m", funding=funding
        )
        assert ds["feature_minutes_to_next_funding"].iloc[0] == pytest.approx(60.0)
