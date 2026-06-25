from __future__ import annotations

import hashlib
import json
from typing import Any, cast

import pandas as pd

DEFAULT_STOP_PCT = 0.005
DEFAULT_TAKE_PCT = 0.01
ATR_STOP_MULTIPLIER = 1.0
ATR_TAKE_MULTIPLIER = 2.0


def generate_mean_reversion_signals(features: pd.DataFrame, symbol: str) -> pd.DataFrame:
    rows = []
    for row in features.dropna(subset=["feature_zscore_close_20"]).itertuples(index=False):
        row_any = cast(Any, row)
        zscore = float(row_any.feature_zscore_close_20)
        event_time = row_any.event_time
        close_vs_ema = float(getattr(row_any, "feature_close_vs_ema_20", 0.0))
        entry_ref = float(getattr(row_any, "close", 0.0) or 0.0)
        atr = float(getattr(row_any, "feature_atr_14", float("nan")))
        side = "no_trade"
        if zscore < -2.0:
            side = "long"
        elif zscore > 2.0:
            side = "short"

        stop_ref = 0.0
        take_ref = 0.0
        if entry_ref > 0.0 and side in {"long", "short"}:
            atr_valid = pd.notna(atr) and atr > 0.0
            if atr_valid:
                stop_distance = atr * ATR_STOP_MULTIPLIER
                take_distance = atr * ATR_TAKE_MULTIPLIER
            else:
                stop_distance = entry_ref * DEFAULT_STOP_PCT
                take_distance = entry_ref * DEFAULT_TAKE_PCT

            if side == "long":
                stop_ref = max(entry_ref - stop_distance, entry_ref * 0.0001)
                take_ref = entry_ref + take_distance
            else:
                stop_ref = entry_ref + stop_distance
                take_ref = max(entry_ref - take_distance, entry_ref * 0.0001)

        snapshot = {"zscore": zscore, "close_vs_ema": close_vs_ema}
        rows.append(
            {
                "signal_id": hashlib.sha256(f"mr-{symbol}-{event_time}".encode()).hexdigest()[:24],
                "symbol": symbol,
                "event_time": event_time,
                "strategy_name": "mean_reversion",
                "side": side,
                "score": abs(zscore),
                "confidence": min(abs(zscore) / 4.0, 1.0),
                "entry_ref": entry_ref,
                "stop_ref": stop_ref,
                "take_ref": take_ref,
                "ttl_sec": 1800,
                "rationale_json": {"reason": "rolling z-score mean reversion", "zscore": zscore},
                "feature_snapshot_hash": hashlib.sha256(json.dumps(snapshot).encode()).hexdigest(),
                "risk_status": "accepted",
                "risk_rejection_reason": "",
            }
        )
    return pd.DataFrame(rows)
