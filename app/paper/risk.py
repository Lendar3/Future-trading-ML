from __future__ import annotations

from dataclasses import dataclass

from app.core.enums import RiskStatus, SignalSide
from app.core.types import Signal


@dataclass(frozen=True)
class RiskLimits:
    max_risk_per_trade: float = 0.005
    max_daily_loss: float = 0.02
    max_drawdown: float = 0.1
    max_open_positions: int = 3
    max_leverage: float = 3.0


@dataclass(frozen=True)
class RiskState:
    equity: float
    daily_pnl: float = 0.0
    max_drawdown: float = 0.0
    open_positions: int = 0
    leverage: float = 1.0
    kill_switch_active: bool = False


def check_signal_risk(
    signal: Signal, state: RiskState, limits: RiskLimits
) -> tuple[RiskStatus, str]:
    if signal.side == SignalSide.NO_TRADE:
        return RiskStatus.REJECTED, "no_trade signal"
    if state.kill_switch_active:
        return RiskStatus.REJECTED, "kill switch active"
    if state.daily_pnl <= -limits.max_daily_loss * state.equity:
        return RiskStatus.REJECTED, "daily loss limit reached"
    if state.max_drawdown >= limits.max_drawdown:
        return RiskStatus.REJECTED, "max drawdown limit reached"
    if state.open_positions >= limits.max_open_positions:
        return RiskStatus.REJECTED, "max open positions reached"
    if state.leverage > limits.max_leverage:
        return RiskStatus.REJECTED, "max leverage exceeded"
    if signal.entry_ref <= 0:
        return RiskStatus.REJECTED, "missing entry reference"
    if signal.stop_ref <= 0:
        return RiskStatus.REJECTED, "missing stop reference"
    if signal.take_ref <= 0:
        return RiskStatus.REJECTED, "missing take reference"
    if signal.side == SignalSide.LONG and not (
        signal.stop_ref < signal.entry_ref < signal.take_ref
    ):
        return RiskStatus.REJECTED, "invalid long stop/take geometry"
    if signal.side == SignalSide.SHORT and not (
        signal.take_ref < signal.entry_ref < signal.stop_ref
    ):
        return RiskStatus.REJECTED, "invalid short stop/take geometry"
    return RiskStatus.ACCEPTED, ""
