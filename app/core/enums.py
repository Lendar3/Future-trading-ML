from __future__ import annotations

from enum import StrEnum


class SystemMode(StrEnum):
    RESEARCH = "research"
    BACKTEST = "backtest"
    DEMO = "demo"
    PAPER = "paper"
    REAL_ADVISORY = "real_advisory"
    LIVE_DISABLED = "live_disabled"


class SignalSide(StrEnum):
    LONG = "long"
    SHORT = "short"
    NO_TRADE = "no_trade"


class OrderStatus(StrEnum):
    CREATED = "created"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class RiskStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
