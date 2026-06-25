from __future__ import annotations

import pytest

from app.paper.exit_policy import simulate_paper_exit_from_bars


def _fill() -> dict[str, object]:
    return {
        "fill_id": "fill_001",
        "side": "buy_open_long",
        "entry_time": "2026-01-01T00:00:00+00:00",
        "filled_at": "2026-01-01T00:00:00+00:00",
    }


def _signal_long() -> dict[str, object]:
    return {
        "side": "long",
        "entry_ref": 100.0,
        "stop_ref": 95.0,
        "take_ref": 110.0,
        "ttl_sec": 600,
        "event_time": "2026-01-01T00:00:00+00:00",
    }


def _signal_short() -> dict[str, object]:
    return {
        "side": "short",
        "entry_ref": 100.0,
        "stop_ref": 105.0,
        "take_ref": 90.0,
        "ttl_sec": 600,
        "event_time": "2026-01-01T00:00:00+00:00",
    }


def test_exit_policy_long_take_profit() -> None:
    result = simulate_paper_exit_from_bars(
        fill_payload=_fill(),
        signal_payload=_signal_long(),
        ohlc_bars=[
            {
                "event_time": "2026-01-01T00:05:00+00:00",
                "high": 111.0,
                "low": 99.0,
                "close": 105.0,
            }
        ],
    )

    assert result["exit_reason"] == "take_profit"
    assert result["exit_price"] == 110.0


def test_exit_policy_long_stop_loss() -> None:
    result = simulate_paper_exit_from_bars(
        fill_payload=_fill(),
        signal_payload=_signal_long(),
        ohlc_bars=[
            {
                "event_time": "2026-01-01T00:05:00+00:00",
                "high": 101.0,
                "low": 94.0,
                "close": 99.0,
            }
        ],
    )

    assert result["exit_reason"] == "stop_loss"
    assert result["exit_price"] == 95.0


def test_exit_policy_long_same_bar_conflict_prefers_stop_loss() -> None:
    result = simulate_paper_exit_from_bars(
        fill_payload=_fill(),
        signal_payload=_signal_long(),
        ohlc_bars=[
            {
                "event_time": "2026-01-01T00:05:00+00:00",
                "high": 111.0,
                "low": 94.0,
                "close": 108.0,
            }
        ],
    )

    assert result["exit_reason"] == "stop_loss"
    assert result["exit_price"] == 95.0


def test_exit_policy_short_take_profit() -> None:
    result = simulate_paper_exit_from_bars(
        fill_payload=_fill(),
        signal_payload=_signal_short(),
        ohlc_bars=[
            {
                "event_time": "2026-01-01T00:05:00+00:00",
                "high": 101.0,
                "low": 89.0,
                "close": 93.0,
            }
        ],
    )

    assert result["exit_reason"] == "take_profit"
    assert result["exit_price"] == 90.0


def test_exit_policy_short_stop_loss() -> None:
    result = simulate_paper_exit_from_bars(
        fill_payload=_fill(),
        signal_payload=_signal_short(),
        ohlc_bars=[
            {
                "event_time": "2026-01-01T00:05:00+00:00",
                "high": 106.0,
                "low": 96.0,
                "close": 103.0,
            }
        ],
    )

    assert result["exit_reason"] == "stop_loss"
    assert result["exit_price"] == 105.0


def test_exit_policy_short_same_bar_conflict_prefers_stop_loss() -> None:
    result = simulate_paper_exit_from_bars(
        fill_payload=_fill(),
        signal_payload=_signal_short(),
        ohlc_bars=[
            {
                "event_time": "2026-01-01T00:05:00+00:00",
                "high": 106.0,
                "low": 89.0,
                "close": 95.0,
            }
        ],
    )

    assert result["exit_reason"] == "stop_loss"
    assert result["exit_price"] == 105.0


def test_exit_policy_ttl_exit() -> None:
    result = simulate_paper_exit_from_bars(
        fill_payload=_fill(),
        signal_payload=_signal_long(),
        ohlc_bars=[
            {
                "event_time": "2026-01-01T00:05:00+00:00",
                "high": 109.0,
                "low": 96.0,
                "close": 101.0,
            },
            {
                "event_time": "2026-01-01T00:10:00+00:00",
                "high": 112.0,
                "low": 97.0,
                "close": 104.0,
            },
        ],
    )

    assert result["exit_reason"] == "ttl_exit"
    assert result["exit_price"] == 104.0
    assert result["exit_time"] == "2026-01-01T00:10:00+00:00"


def test_exit_policy_no_exit_data() -> None:
    result = simulate_paper_exit_from_bars(
        fill_payload=_fill(),
        signal_payload=_signal_long(),
        ohlc_bars=[],
    )

    assert result["exit_reason"] == "no_exit_data"
    assert result["exit_price"] == 100.0
    assert result["bars_checked"] == 0


def test_exit_policy_invalid_geometry_rejected() -> None:
    bad_signal = {
        **_signal_long(),
        "stop_ref": 120.0,
        "take_ref": 110.0,
    }

    with pytest.raises(ValueError, match="Invalid long geometry"):
        simulate_paper_exit_from_bars(
            fill_payload=_fill(),
            signal_payload=bad_signal,
            ohlc_bars=[],
        )


def test_exit_policy_invalid_side_rejected() -> None:
    bad_signal = {
        **_signal_long(),
        "side": "flat",
    }

    with pytest.raises(ValueError, match="side must be long or short"):
        simulate_paper_exit_from_bars(
            fill_payload=_fill(),
            signal_payload=bad_signal,
            ohlc_bars=[],
        )
