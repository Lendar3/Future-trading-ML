from __future__ import annotations

import pytest

from app.core.modes import ModeState
from app.core.safety import LiveTradingDisabledError
from app.execution.real_disabled import RealBrokerInterface


def test_real_broker_place_order_is_disabled() -> None:
    broker = RealBrokerInterface()
    with pytest.raises(LiveTradingDisabledError):
        broker.place_order(symbol="BTCUSDT", side="buy", quantity=1)


def test_real_broker_cancel_order_is_disabled() -> None:
    broker = RealBrokerInterface()
    with pytest.raises(LiveTradingDisabledError):
        broker.cancel_order(order_id="abc")


def test_mode_state_rejects_real_trading_enabled() -> None:
    with pytest.raises(LiveTradingDisabledError):
        ModeState(real_trading_enabled=True).assert_no_live_trading()
