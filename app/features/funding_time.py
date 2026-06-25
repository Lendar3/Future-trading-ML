"""Funding schedule utilities for Bybit perpetual futures.

Bybit perpetuals settle funding at three fixed UTC times per day:

    00:00 UTC  |  08:00 UTC  |  16:00 UTC

No-lookahead note
-----------------
``minutes_to_next_funding`` is derived solely from the **event_time clock**
and the known funding schedule — it does **not** use any future market data.
The funding schedule is public and fixed, so computing it from event_time is
safe and produces no lookahead.

The *rate* (``feature_funding_last``) is still subject to the backward ASOF
join and the staleness limit in the dataset builder.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

# Funding window offsets within a day, in seconds from midnight UTC.
_WINDOW_OFFSETS_SECONDS: tuple[int, int, int] = (0, 8 * 3600, 16 * 3600)


def minutes_to_next_funding(event_time: datetime) -> float:
    """Return minutes until the next Bybit funding window from *event_time*.

    Funding windows are at 00:00, 08:00, and 16:00 UTC.  The result is in
    the range ``(0.0, 480.0]``.  A timestamp that falls exactly on a window
    boundary returns ``0.0`` (the next window is now / instantaneously).

    Parameters
    ----------
    event_time:
        A timezone-aware datetime (any tz is accepted; computation is done
        in UTC).

    Returns
    -------
    float
        Minutes to the next funding settlement.
    """
    # Normalise to UTC.
    if event_time.tzinfo is None:
        raise ValueError("event_time must be timezone-aware.")
    utc_dt = event_time.astimezone(UTC)

    # Seconds elapsed since midnight UTC for this day.
    midnight = utc_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    elapsed_s = (utc_dt - midnight).total_seconds()

    # Find the next window offset (strictly after elapsed_s, wrapping to
    # the following day).
    best: float | None = None
    for offset_s in _WINDOW_OFFSETS_SECONDS:
        remaining = offset_s - elapsed_s
        if remaining >= 0 and (best is None or remaining < best):
            best = remaining

    # All windows are in the past → wrap to next-day 00:00.
    if best is None:
        best = 86400.0 - elapsed_s  # seconds to next midnight

    return best / 60.0


def minutes_to_next_funding_series(event_times: pd.Series) -> pd.Series:
    """Vectorised version of :func:`minutes_to_next_funding` for a Series.

    Parameters
    ----------
    event_times:
        A Series of timezone-aware :class:`~pandas.Timestamp` values.

    Returns
    -------
    pd.Series
        Float series aligned to the input index.
    """
    return event_times.apply(lambda t: minutes_to_next_funding(t.to_pydatetime()))
