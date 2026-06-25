from __future__ import annotations

APP_TO_BYBIT_INTERVAL = {
    "1m": "1",
    "5m": "5",
    "15m": "15",
    "1h": "60",
}

BYBIT_INTERVAL_TO_APP = {value: key for key, value in APP_TO_BYBIT_INTERVAL.items()}


def normalize_app_timeframe(value: str) -> str:
    timeframe = value.strip()
    if timeframe in APP_TO_BYBIT_INTERVAL:
        return timeframe
    if timeframe in BYBIT_INTERVAL_TO_APP:
        return BYBIT_INTERVAL_TO_APP[timeframe]
    raise ValueError(
        f"Unsupported app timeframe: {value!r}. "
        f"Expected one of {sorted(APP_TO_BYBIT_INTERVAL)}."
    )


def app_timeframe_to_bybit_interval(value: str) -> str:
    return APP_TO_BYBIT_INTERVAL[normalize_app_timeframe(value)]
