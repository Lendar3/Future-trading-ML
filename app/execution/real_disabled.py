from __future__ import annotations

from typing import Any

from app.core.safety import raise_live_trading_disabled
from app.execution.base import BrokerInterface


class RealBrokerInterface(BrokerInterface):
    def place_order(self, *args: Any, **kwargs: Any) -> Any:
        raise_live_trading_disabled()

    def cancel_order(self, *args: Any, **kwargs: Any) -> Any:
        raise_live_trading_disabled()

    def amend_order(self, *args: Any, **kwargs: Any) -> Any:
        raise_live_trading_disabled()

    def close_position(self, *args: Any, **kwargs: Any) -> Any:
        raise_live_trading_disabled()
