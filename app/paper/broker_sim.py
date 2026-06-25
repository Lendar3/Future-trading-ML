from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from app.core.enums import OrderStatus, SignalSide
from app.core.types import Signal


@dataclass(frozen=True)
class PaperOrder:
    order_id: str
    symbol: str
    side: SignalSide
    quantity: float
    status: OrderStatus
    created_at: datetime


@dataclass(frozen=True)
class PaperFill:
    fill_id: str
    order_id: str
    symbol: str
    side: SignalSide
    quantity: float
    price: float
    fee: float
    filled_at: datetime


class PaperBrokerSim:
    def create_order(self, signal: Signal, quantity: float, created_at: datetime) -> PaperOrder:
        if signal.side == SignalSide.NO_TRADE:
            return PaperOrder(
                str(uuid4()),
                signal.symbol,
                signal.side,
                0.0,
                OrderStatus.REJECTED,
                created_at,
            )
        return PaperOrder(
            str(uuid4()), signal.symbol, signal.side, quantity, OrderStatus.CREATED, created_at
        )

    def fill_order(self, order: PaperOrder, price: float, filled_at: datetime) -> PaperFill:
        fee = abs(order.quantity * price) * 0.00055
        return PaperFill(
            str(uuid4()),
            order.order_id,
            order.symbol,
            order.side,
            order.quantity,
            price,
            fee,
            filled_at,
        )
