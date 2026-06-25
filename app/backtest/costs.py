from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC

import pandas as pd


@dataclass(frozen=True)
class CostAssumptions:
    maker_fee_bp: float = 2.0
    taker_fee_bp: float = 5.5
    spread_bp: float = 1.0
    slippage_bp: float = 2.0
    safety_margin_bp: float = 2.0
    funding_bp: float = 0.0

    def total_round_trip_bp(self, taker: bool = True) -> float:
        fee = self.taker_fee_bp if taker else self.maker_fee_bp
        return 2 * fee + self.spread_bp + self.slippage_bp + self.funding_bp

    def min_edge_required_bp(self, taker: bool = True) -> float:
        return self.total_round_trip_bp(taker=taker) + self.safety_margin_bp


def bp_to_return(bp: float) -> float:
    return bp / 10_000


def return_to_bp(value: float) -> float:
    return value * 10_000


FUNDING_WINDOW_HOURS_UTC: tuple[int, ...] = (0, 8, 16)


def _to_utc_timestamp(value: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize(UTC)
    return ts.tz_convert(UTC)


def count_funding_events(entry_time: pd.Timestamp, exit_time: pd.Timestamp) -> int:
    start = _to_utc_timestamp(entry_time)
    end = _to_utc_timestamp(exit_time)
    if end <= start:
        return 0

    current_day = start.normalize()
    final_day = end.normalize()
    events = 0
    while current_day <= final_day:
        for hour in FUNDING_WINDOW_HOURS_UTC:
            funding_ts = current_day + pd.Timedelta(hours=hour)
            if start < funding_ts <= end:
                events += 1
        current_day += pd.Timedelta(days=1)
    return events


def funding_for_holding_window(
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    side: str,
    funding_rate: float,
) -> tuple[int, float, float]:
    events = count_funding_events(entry_time=entry_time, exit_time=exit_time)
    if events == 0 or funding_rate == 0.0:
        return 0, 0.0, 0.0

    if side == "long":
        direction = 1
    elif side == "short":
        direction = -1
    else:
        raise ValueError(f"Unsupported side for funding calculation: {side}")

    # Positive funding_rate means longs pay and shorts receive.
    funding_return = float(events) * (-direction * funding_rate)
    funding_bp = return_to_bp(-funding_return)
    return events, funding_return, funding_bp
