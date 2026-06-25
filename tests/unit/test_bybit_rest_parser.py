from __future__ import annotations

from datetime import UTC, datetime

from app.core.timeframes import app_timeframe_to_bybit_interval, normalize_app_timeframe
from app.ingestion.bybit_rest import (
    interval_to_milliseconds,
    parse_funding_item,
    parse_kline_row,
    parse_long_short_ratio_item,
    parse_open_interest_item,
    raw_bar_to_upsert_row,
    utc_from_ms,
    utc_ms,
)


def test_app_timeframe_maps_to_bybit_interval_without_mixing_storage_format() -> None:
    assert normalize_app_timeframe("1m") == "1m"
    assert normalize_app_timeframe("1") == "1m"
    assert app_timeframe_to_bybit_interval("1m") == "1"
    assert app_timeframe_to_bybit_interval("5m") == "5"
    assert app_timeframe_to_bybit_interval("15m") == "15"
    assert app_timeframe_to_bybit_interval("1h") == "60"


def test_utc_timestamp_helpers_are_timezone_aware() -> None:
    value = utc_from_ms("1778264400000")
    assert value.tzinfo == UTC
    assert utc_ms(value) == 1778264400000


def test_interval_to_milliseconds_supports_bybit_intervals() -> None:
    assert interval_to_milliseconds("1") == 60_000
    assert interval_to_milliseconds("5") == 300_000
    assert interval_to_milliseconds("D") == 86_400_000


def test_parse_regular_and_mark_kline_rows() -> None:
    fetch_time = datetime(2026, 1, 1, tzinfo=UTC)
    regular = parse_kline_row(
        ["1778264400000", "100", "101", "99", "100.5", "10", "1005"],
        symbol="BTCUSDT",
        timeframe="1",
        fetch_time=fetch_time,
        source="bybit_public",
    )
    mark = parse_kline_row(
        ["1778264400000", "100", "101", "99", "100.5"],
        symbol="BTCUSDT",
        timeframe="1",
        fetch_time=fetch_time,
        source="bybit_public",
    )
    assert regular.volume == 10.0
    assert mark.volume == 0.0
    assert raw_bar_to_upsert_row(regular)["metadata_json"]


def test_parse_public_market_points() -> None:
    fetch_time = datetime(2026, 1, 1, tzinfo=UTC)
    funding = parse_funding_item(
        {"symbol": "BTCUSDT", "fundingRate": "0.0001", "fundingRateTimestamp": "1778264400000"},
        fetch_time,
        "bybit_public",
    )
    oi = parse_open_interest_item(
        {"openInterest": "12345.5", "timestamp": "1778264400000"},
        "BTCUSDT",
        fetch_time,
        "bybit_public",
    )
    ratio = parse_long_short_ratio_item(
        {"buyRatio": "0.55", "sellRatio": "0.45", "timestamp": "1778264400000"},
        "BTCUSDT",
        fetch_time,
        "bybit_public",
    )
    assert funding["funding_rate"] == 0.0001
    assert oi["open_interest"] == 12345.5
    assert ratio["long_short_ratio"] == 0.55 / 0.45
