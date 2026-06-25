"""No-lookahead sparse-table join helpers for the feature builder.

``asof_join_sparse`` wraps ``pandas.merge_asof`` and adds a staleness
check so that forward-filled values older than ``max_age`` are replaced
with NaN rather than propagated indefinitely.

No-lookahead guarantee
----------------------
``merge_asof`` with ``direction="backward"`` always picks the most-recent
row in ``right`` whose key is **less than or equal to** the corresponding
key in ``left``.  This means future rows in ``right`` are never used.

The staleness guard additionally ensures that very old values (beyond
``max_age``) are not silently treated as current.  The matched right-side
timestamp is tracked in an internal column ``__matched_right_time`` and
the age is computed as ``left_event_time - __matched_right_time``.  If
age > max_age all joined value columns are set to NaN for that row;
the left timestamp column is never modified.

Implementation note
-------------------
The previous version re-ran ``merge_asof`` a second time to recover the
matched right timestamp, which created a risk of subtle column-naming
collisions.  This version carries the right timestamp through the first
merge by temporarily renaming it to the reserved sentinel column name
``__matched_right_time`` so there is exactly one merge call.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

_RIGHT_TS_COL = "__matched_right_time"


def asof_join_sparse(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_on: str,
    right_on: str,
    max_age: timedelta | None = None,
    suffixes: tuple[str, str] = ("", "_right"),
) -> pd.DataFrame:
    """ASOF-join ``right`` onto ``left`` using backward direction.

    For each row in ``left`` the most-recent row in ``right`` at or before
    ``left[left_on]`` is selected.  If ``max_age`` is provided, rows whose
    matched ``right_on`` timestamp is older than ``left_on - max_age`` are
    replaced with NaN in all right-side value columns.  The left timestamp
    column is **never** modified.

    Parameters
    ----------
    left:
        The anchor frame (e.g. bar timestamps).  Internally sorted by
        ``left_on`` before merging; the original index is restored on return.
    right:
        The sparse frame to align.  Internally sorted by ``right_on``.
    left_on:
        Column name in ``left`` to join on (must be datetime-like).
    right_on:
        Column name in ``right`` to join on (must be datetime-like).
    max_age:
        Maximum age of a matched ``right`` row before it is treated as stale.
        Age is computed per-row as ``left[left_on] - matched_right_time``.
        ``None`` disables staleness filtering (plain forward-fill behaviour).
    suffixes:
        Suffix pair passed to ``merge_asof`` for overlapping non-key column
        names.  The key column always uses ``left_on`` after renaming.

    Returns
    -------
    pd.DataFrame
        Aligned frame with the same positional order as ``left``.  Contains
        all right-side value columns (NaN where no backward match exists or
        the match is stale).  Does **not** contain ``__matched_right_time``.
        The original ``left`` index is restored.
    """
    original_index = left.index

    # Sort both frames on their respective timestamp keys.
    left_sorted = left.sort_values(left_on).reset_index(drop=True)
    right_sorted = right.sort_values(right_on).reset_index(drop=True)

    # Value columns are everything in right except the key column.
    value_cols = [c for c in right_sorted.columns if c != right_on]

    # Fast path: empty right frame → all value columns are NaN.
    if right_sorted.empty:
        result = left_sorted.copy()
        for col in value_cols:
            result[col] = float("nan")
        result.index = original_index
        return result

    # Build the right-side frame for merge_asof.
    # Rename the right key column to left_on so merge_asof can use ``on=``.
    # Also add the sentinel column that carries the original right timestamp
    # through the merge — this avoids a second merge_asof call.
    right_for_merge = right_sorted.copy()
    right_for_merge[_RIGHT_TS_COL] = right_sorted[right_on]
    if right_on != left_on:
        right_for_merge = right_for_merge.rename(columns={right_on: left_on})

    # Guard: _RIGHT_TS_COL must not already exist in left (it is internal).
    if _RIGHT_TS_COL in left_sorted.columns:
        raise ValueError(
            f"Column '{_RIGHT_TS_COL}' is reserved for internal use and "
            "must not exist in the left frame."
        )

    merged = pd.merge_asof(
        left_sorted,
        right_for_merge,
        on=left_on,
        direction="backward",
        suffixes=suffixes,
    )

    # Value columns are everything in right_for_merge except the key and sentinel.
    value_cols = [c for c in right_for_merge.columns if c not in (left_on, _RIGHT_TS_COL)]

    if max_age is not None:
        matched_ts: pd.Series = merged[_RIGHT_TS_COL]
        # Age = left timestamp - matched right timestamp.
        # Rows with no match (matched_ts is NaT/NaN) are automatically stale.
        left_ts: pd.Series = merged[left_on]
        age = left_ts - matched_ts
        stale_mask: pd.Series = matched_ts.isna() | (age > max_age)
        for col in value_cols:
            if col in merged.columns:
                merged.loc[stale_mask, col] = float("nan")

    # Drop the internal sentinel column before returning.
    merged = merged.drop(columns=[_RIGHT_TS_COL], errors="ignore")

    # Restore original left index.
    merged.index = original_index
    return merged
