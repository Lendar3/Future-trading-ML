"""Dataset quality report.

``DatasetQualityReport`` is a lightweight dataclass computed from any
feature/label DataFrame produced by the dataset builder.  It is
intentionally dependency-free (no DB, no API) so it can be called from
pipelines, unit tests, and API routes alike.

The report covers:

* Row count and feature/label column counts.
* NaN ratio per feature group (price/vol, basis, funding, OI, LS).
* Overall NaN ratio across all feature columns.
* First and last event_time.
* feature_set_name, symbol, timeframe (read from the dataset's metadata
  columns if present, else ``"unknown"``).
* Label distribution (counts + percentages per class).
* Number of rows dropped due to missing labels.
* Stale sparse feature count: rows where a sparse feature group has all NaN
  values (approximation of staleness coverage without requiring the original
  join timestamps).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Feature group prefix maps (mirrors routes_ui.py grouping logic)
# ---------------------------------------------------------------------------

_GROUP_PREFIXES: dict[str, tuple[str, ...]] = {
    "price_volume_volatility": (
        "feature_ret",
        "feature_log",
        "feature_close",
        "feature_zscore",
        "feature_distance",
        "feature_volume",
        "feature_vol",
        "feature_atr",
        "feature_bb",
        "feature_hv",
    ),
    "basis": (
        "feature_mark",
        "feature_index",
        "feature_basis",
        "feature_premium",
    ),
    "funding": (
        "feature_funding",
        "feature_minutes_to_next",
    ),
    "open_interest": (
        "feature_open_interest",
        "feature_oi",
        "feature_price_up_oi",
        "feature_price_down_oi",
        "feature_squeeze",
    ),
    "long_short": (
        "feature_long_short",
        "feature_ls",
        "feature_long_account",
        "feature_short_account",
        "feature_crowded",
    ),
}


def _classify_feature(col: str) -> str:
    for group, prefixes in _GROUP_PREFIXES.items():
        if any(col.startswith(p) for p in prefixes):
            return group
    return "other"


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass
class DatasetQualityReport:
    """Immutable snapshot of dataset quality metrics."""

    symbol: str
    timeframe: str
    feature_set_name: str

    total_rows: int
    feature_columns: int
    label_columns: int

    first_event_time: str  # ISO-8601 string
    last_event_time: str

    # NaN ratio (0.0–1.0) per feature group; groups absent → 0.0.
    nan_ratio_by_group: dict[str, float] = field(default_factory=dict)
    # Overall NaN ratio across all feature columns.
    overall_nan_ratio: float = 0.0

    # Label distribution: {label_value: count}.
    label_distribution: dict[str, int] = field(default_factory=dict)
    # Label distribution as percentage: {label_value: pct}.
    label_distribution_pct: dict[str, float] = field(default_factory=dict)
    # Rows that had no label (before label-drop; computed on the raw dataset).
    rows_missing_labels: int = 0

    # Per-group: count of rows where ALL features in that group are NaN.
    stale_sparse_rows: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "feature_set_name": self.feature_set_name,
            "total_rows": self.total_rows,
            "feature_columns": self.feature_columns,
            "label_columns": self.label_columns,
            "first_event_time": self.first_event_time,
            "last_event_time": self.last_event_time,
            "nan_ratio_by_group": self.nan_ratio_by_group,
            "overall_nan_ratio": self.overall_nan_ratio,
            "label_distribution": self.label_distribution,
            "label_distribution_pct": self.label_distribution_pct,
            "rows_missing_labels": self.rows_missing_labels,
            "stale_sparse_rows": self.stale_sparse_rows,
        }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def compute_dataset_quality(dataset: pd.DataFrame) -> DatasetQualityReport:
    """Compute a :class:`DatasetQualityReport` from a feature/label DataFrame.

    The DataFrame is expected to have the columns produced by
    ``build_dataset_from_bars`` or ``build_dataset_from_public_tables``:
    columns prefixed ``feature_``, ``label_``, plus metadata columns
    ``symbol``, ``timeframe``, ``feature_set_name``, ``event_time``.

    Parameters
    ----------
    dataset:
        Output of a dataset builder function.  Must contain ``event_time``.

    Returns
    -------
    DatasetQualityReport
    """
    feature_cols = [c for c in dataset.columns if c.startswith("feature_")]
    label_cols = [c for c in dataset.columns if c.startswith("label_")]

    # ---- Metadata --------------------------------------------------------
    symbol = str(dataset["symbol"].iloc[0]) if "symbol" in dataset.columns else "unknown"
    timeframe = str(dataset["timeframe"].iloc[0]) if "timeframe" in dataset.columns else "unknown"
    feature_set_name = (
        str(dataset["feature_set_name"].iloc[0])
        if "feature_set_name" in dataset.columns
        else "unknown"
    )

    event_times = pd.to_datetime(dataset["event_time"])
    first_et = event_times.min()
    last_et = event_times.max()

    # ---- NaN ratios ------------------------------------------------------
    n = len(dataset)
    groups_present: dict[str, list[str]] = {g: [] for g in _GROUP_PREFIXES}
    groups_present["other"] = []
    for col in feature_cols:
        groups_present.setdefault(_classify_feature(col), []).append(col)

    nan_ratio_by_group: dict[str, float] = {}
    for group, cols in groups_present.items():
        if cols:
            nan_ratio_by_group[group] = float(dataset[cols].isna().mean().mean())

    overall_nan_ratio = float(dataset[feature_cols].isna().mean().mean()) if feature_cols else 0.0

    # ---- Label distribution ----------------------------------------------
    rows_missing_labels = 0
    label_distribution: dict[str, int] = {}
    label_distribution_pct: dict[str, float] = {}
    if label_cols:
        primary_label = label_cols[0]
        rows_missing_labels = int(dataset[primary_label].isna().sum())
        labeled = dataset[primary_label].dropna()
        if not labeled.empty:
            counts = labeled.value_counts().sort_index()
            label_distribution = {str(k): int(v) for k, v in counts.items()}
            total_labeled = labeled.count()
            label_distribution_pct = {
                str(k): round(float(v) / total_labeled * 100, 2) for k, v in counts.items()
            }

    # ---- Stale sparse rows -----------------------------------------------
    # Sparse groups: basis, funding, OI, LS.
    sparse_groups = ("basis", "funding", "open_interest", "long_short")
    stale_sparse_rows: dict[str, int] = {}
    for group in sparse_groups:
        cols = groups_present.get(group, [])
        if cols:
            all_nan_mask = dataset[cols].isna().all(axis=1)
            stale_sparse_rows[group] = int(all_nan_mask.sum())

    return DatasetQualityReport(
        symbol=symbol,
        timeframe=timeframe,
        feature_set_name=feature_set_name,
        total_rows=n,
        feature_columns=len(feature_cols),
        label_columns=len(label_cols),
        first_event_time=first_et.isoformat() if not pd.isna(first_et) else "",
        last_event_time=last_et.isoformat() if not pd.isna(last_et) else "",
        nan_ratio_by_group=nan_ratio_by_group,
        overall_nan_ratio=overall_nan_ratio,
        label_distribution=label_distribution,
        label_distribution_pct=label_distribution_pct,
        rows_missing_labels=rows_missing_labels,
        stale_sparse_rows=stale_sparse_rows,
    )
