"""Dataset builder — v1.6: enriched with real public Bybit market features.

Feature-set naming
------------------
* ``"v1"`` — legacy OHLCV-only path (``build_dataset_from_bars``).
* ``"v1.6_public"`` — enriched path (``build_dataset_from_public_tables``).

No-lookahead guarantees
-----------------------
* ``bars`` are sorted by ``event_time`` before any computation.
* Mark and index price bars are joined on exact ``event_time``.
* Funding, OI, and long-short ratio are sparse tables joined via
  ``asof_join_sparse`` with ``direction="backward"`` so each bar row gets
  only the most-recent known value **at or before** its ``event_time``.
* A ``max_age_seconds`` parameter enforces a staleness limit: values older
  than the limit are set to NaN.
* ``source_available_time`` is computed per-row as the maximum available
  timestamp among all data sources used for that row.  It is always
  ``<= event_time`` and is validated by ``validate_feature_timestamps``.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from app.backtest.costs import CostAssumptions
from app.features.basis_features import build_basis_features
from app.features.dataset_join import asof_join_sparse
from app.features.funding_time import minutes_to_next_funding_series
from app.features.price_features import build_price_features
from app.features.volatility_features import build_volatility_features
from app.features.volume_features import build_volume_features
from app.models.labels import build_edge_labels
from app.models.validation import validate_feature_timestamps

# Default staleness limits for sparse tables.
FUNDING_MAX_AGE_SECONDS: int = 8 * 3600 + 600  # 8 h 10 min (one period + grace)
OI_MAX_AGE_SECONDS: int = 900  # 15 minutes
LS_MAX_AGE_SECONDS: int = 900  # 15 minutes

FEATURE_SET_LEGACY = "v1"
FEATURE_SET_PUBLIC = "v1.6_public"


def _time_delta_feature(
    *,
    event_time: pd.Series,
    values: pd.Series,
    window: timedelta,
) -> pd.Series:
    """Delta against the latest value at-or-before (event_time - window)."""
    if values.empty:
        return pd.Series(dtype="float64", index=values.index)

    event_ns = (
        pd.to_datetime(event_time, utc=True)
        .dt.tz_localize(None)
        .astype("datetime64[ns]")
        .astype("int64")
        .to_numpy()
    )
    window_ns = pd.Timedelta(window).value
    target_ns = event_ns - window_ns
    value_arr = values.astype("float64").to_numpy()

    # For each t_i find largest j where event_time_j <= t_i - window.
    idx = event_ns.searchsorted(target_ns, side="right") - 1
    out = pd.Series(float("nan"), index=values.index, dtype="float64")

    for i, j in enumerate(idx):
        if j < 0:
            continue
        cur = value_arr[i]
        prev = value_arr[j]
        if pd.isna(cur) or pd.isna(prev):
            continue
        out.iat[i] = cur - prev
    return out


def build_dataset_from_bars(
    bars: pd.DataFrame,
    symbol: str,
    timeframe: str,
    feature_set_name: str = FEATURE_SET_LEGACY,
    label_horizon: int = 3,
) -> pd.DataFrame:
    """Build a feature/label dataset from OHLCV bars only (legacy path).

    ``source_available_time`` is set equal to ``event_time`` (bar close).
    """
    ordered = bars.sort_values("event_time").copy()
    features = pd.concat(
        [
            build_price_features(ordered),
            build_volume_features(ordered),
            build_volatility_features(ordered),
        ],
        axis=1,
    )
    labels = build_edge_labels(ordered["close"].astype(float), label_horizon, CostAssumptions())
    dataset = pd.concat([features, labels], axis=1)
    dataset["symbol"] = symbol
    dataset["timeframe"] = timeframe
    dataset["feature_set_name"] = feature_set_name
    dataset["event_time"] = ordered["event_time"].to_list()
    dataset["source_available_time"] = ordered["event_time"].to_list()
    dataset["close"] = ordered["close"].astype(float).values
    validate_feature_timestamps(dataset)
    return dataset


def build_dataset_from_public_tables(
    bars: pd.DataFrame,
    symbol: str,
    timeframe: str,
    mark_bars: pd.DataFrame | None = None,
    index_bars: pd.DataFrame | None = None,
    funding: pd.DataFrame | None = None,
    open_interest: pd.DataFrame | None = None,
    long_short_ratio: pd.DataFrame | None = None,
    feature_set_name: str = FEATURE_SET_PUBLIC,
    label_horizon: int = 3,
    funding_max_age_seconds: int = FUNDING_MAX_AGE_SECONDS,
    oi_max_age_seconds: int = OI_MAX_AGE_SECONDS,
    ls_max_age_seconds: int = LS_MAX_AGE_SECONDS,
) -> pd.DataFrame:
    """Build a feature/label dataset enriched with all available public tables.

    Parameters
    ----------
    bars:
        OHLCV bar frame from ``raw_bars``.  Must contain ``event_time``.
    symbol, timeframe:
        Metadata tags written into every row.
    mark_bars, index_bars:
        Optional mark/index price kline frames (same schema as ``bars``).
        Joined on exact ``event_time``; unmatched rows produce NaN basis
        features.
    funding, open_interest, long_short_ratio:
        Optional sparse tables.  Joined via ``asof_join_sparse`` (backward)
        with the corresponding max-age staleness limit.
    feature_set_name:
        Written as ``feature_set_name`` metadata column.  Default:
        ``"v1.6_public"``.
    label_horizon:
        Number of bars ahead for labelling.
    *_max_age_seconds:
        Staleness limit per sparse table.

    source_available_time
    ---------------------
    For each bar row the ``source_available_time`` is the **maximum** known
    data availability timestamp among:

    * The bar's own ``event_time`` (always present).
    * The matched right-side timestamp from each sparse join (NaN if stale /
      not matched).

    Because ``asof_join_sparse`` never uses future rows, and we take the max
    over all sources, this field is always ``<= event_time``.
    """
    ordered = bars.sort_values("event_time").reset_index(drop=True).copy()
    feature_parts: list[pd.DataFrame] = [
        build_price_features(ordered),
        build_volume_features(ordered),
        build_volatility_features(ordered),
    ]

    # source_available_time: conservatively set to event_time for every row.
    # No-lookahead is enforced by asof_join_sparse (backward direction) which
    # never selects right-side rows whose timestamp > left event_time.  The
    # sat == event_time is always valid because validate_feature_timestamps
    # requires sat <= event_time.
    source_available_time: pd.Series = ordered["event_time"].copy()

    # ---- Basis features (mark price − index price) --------------------
    if (
        mark_bars is not None
        and not mark_bars.empty
        and index_bars is not None
        and not index_bars.empty
    ):
        # Deduplicate on event_time so the merge is always 1:1 with ordered.
        # Multiple ingestion sources can produce duplicate event_time rows;
        # keep the last (most-recently inserted) value.
        mark_sorted = (
            mark_bars.sort_values("event_time")
            .drop_duplicates(subset="event_time", keep="last")
            .reset_index(drop=True)
        )
        index_sorted = (
            index_bars.sort_values("event_time")
            .drop_duplicates(subset="event_time", keep="last")
            .reset_index(drop=True)
        )

        # Exact left-join on event_time — guaranteed to produce the same
        # number of rows as ordered after deduplication of the right sides.
        basis_left = (
            ordered[["event_time"]]
            .copy()
            .merge(
                mark_sorted[["event_time", "close"]].rename(columns={"close": "mark_price"}),
                on="event_time",
                how="left",
            )
            .merge(
                index_sorted[["event_time", "close"]].rename(columns={"close": "index_price"}),
                on="event_time",
                how="left",
            )
        )
        # basis_left now has the same number of rows as ordered; reset so
        # the boolean mask below aligns with ordered.index / basis_frame.index.
        basis_left = basis_left.reset_index(drop=True)
        has_basis = basis_left["mark_price"].notna() & basis_left["index_price"].notna()
        if has_basis.any():
            basis_frame = pd.DataFrame(float("nan"), index=ordered.index, columns=[])
            computed = build_basis_features(basis_left[has_basis].reset_index(drop=True))
            basis_positions = [int(index) for index in has_basis[has_basis].index]
            for col in computed.columns:
                col_values = [float("nan")] * len(ordered)
                for position, value in zip(basis_positions, computed[col].to_numpy(), strict=True):
                    col_values[position] = float(value)
                col_vals = pd.Series(col_values, index=ordered.index, dtype="float64")
                basis_frame[col] = col_vals
            feature_parts.append(basis_frame)

    # ---- Funding features (sparse ASOF, backward) ---------------------
    if funding is not None and not funding.empty:
        funding_aligned = asof_join_sparse(
            left=ordered[["event_time"]],
            right=funding[["event_time", "funding_rate"]],
            left_on="event_time",
            right_on="event_time",
            max_age=timedelta(seconds=funding_max_age_seconds),
        )
        # feature_funding_last: latest known rate at or before event_time
        # (NaN when no fresh rate is available within max_age).
        # feature_minutes_to_next_funding: derived from event_time only via
        # the known public Bybit schedule (00:00 / 08:00 / 16:00 UTC).
        # This is NOT lookahead — the schedule is fixed and public.
        funding_feature_frame = pd.DataFrame(index=ordered.index)
        funding_feature_frame["feature_funding_last"] = funding_aligned["funding_rate"].values
        funding_feature_frame["feature_minutes_to_next_funding"] = minutes_to_next_funding_series(
            ordered["event_time"]
        ).values
        feature_parts.append(funding_feature_frame)

    # ---- Open Interest features (sparse ASOF, backward) ---------------
    if open_interest is not None and not open_interest.empty:
        oi_aligned = asof_join_sparse(
            left=ordered[["event_time"]],
            right=open_interest[["event_time", "open_interest"]],
            left_on="event_time",
            right_on="event_time",
            max_age=timedelta(seconds=oi_max_age_seconds),
        )
        oi_series = oi_aligned["open_interest"]
        oi_feature_frame = pd.DataFrame(index=ordered.index)
        oi_feature_frame["feature_open_interest"] = oi_series
        oi_feature_frame["feature_oi_delta_5m"] = _time_delta_feature(
            event_time=ordered["event_time"],
            values=oi_series,
            window=timedelta(minutes=5),
        )
        oi_feature_frame["feature_oi_delta_1h"] = _time_delta_feature(
            event_time=ordered["event_time"],
            values=oi_series,
            window=timedelta(hours=1),
        )
        feature_parts.append(oi_feature_frame)

    # ---- Long-short ratio features (sparse ASOF, backward) ------------
    if long_short_ratio is not None and not long_short_ratio.empty:
        ls_aligned = asof_join_sparse(
            left=ordered[["event_time"]],
            right=long_short_ratio[["event_time", "long_short_ratio"]],
            left_on="event_time",
            right_on="event_time",
            max_age=timedelta(seconds=ls_max_age_seconds),
        )
        ls_series = ls_aligned["long_short_ratio"]
        ls_feature_frame = pd.DataFrame(index=ordered.index)
        ls_feature_frame["feature_long_short_ratio"] = ls_series
        ls_feature_frame["feature_ls_ratio_change_1h"] = _time_delta_feature(
            event_time=ordered["event_time"],
            values=ls_series,
            window=timedelta(hours=1),
        )
        feature_parts.append(ls_feature_frame)

    features = pd.concat(feature_parts, axis=1)
    labels = build_edge_labels(ordered["close"].astype(float), label_horizon, CostAssumptions())
    dataset = pd.concat([features, labels], axis=1)
    dataset["symbol"] = symbol
    dataset["timeframe"] = timeframe
    dataset["feature_set_name"] = feature_set_name
    dataset["event_time"] = ordered["event_time"].to_list()
    # source_available_time uses to_list() to preserve tz-aware datetimes
    # (same pattern as event_time above).  Using .values would strip tz
    # and cause InvalidComparison in validate_feature_timestamps.
    dataset["source_available_time"] = source_available_time.to_list()
    dataset["close"] = ordered["close"].astype(float).values
    validate_feature_timestamps(dataset)
    return dataset
