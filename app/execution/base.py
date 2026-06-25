from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BrokerInterface(ABC):
    @abstractmethod
    def place_order(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
