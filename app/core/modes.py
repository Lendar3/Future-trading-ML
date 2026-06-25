from __future__ import annotations

from dataclasses import dataclass

from app.core.enums import SystemMode
from app.core.safety import LiveTradingDisabledError


@dataclass(frozen=True)
class ModeState:
    system_mode: SystemMode = SystemMode.RESEARCH
    real_trading_enabled: bool = False
    demo_trading_enabled: bool = False
    paper_trading_enabled: bool = False

    def can_create_orders(self) -> bool:
        return self.system_mode in {SystemMode.DEMO, SystemMode.PAPER} and (
            self.demo_trading_enabled or self.paper_trading_enabled
        )

    def assert_no_live_trading(self) -> None:
        if self.real_trading_enabled:
            raise LiveTradingDisabledError("Real trading cannot be enabled in v1.")


def transition_mode(current: ModeState, next_mode: SystemMode) -> ModeState:
    if next_mode == SystemMode.REAL_ADVISORY:
        return ModeState(system_mode=next_mode, real_trading_enabled=False)
    if next_mode == SystemMode.DEMO:
        return ModeState(system_mode=next_mode, demo_trading_enabled=True)
    if next_mode == SystemMode.PAPER:
        return ModeState(system_mode=next_mode, paper_trading_enabled=True)
    if next_mode == SystemMode.LIVE_DISABLED:
        return ModeState(system_mode=next_mode)
    return ModeState(system_mode=next_mode)
