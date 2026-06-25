from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RiskConfig:
    max_risk_per_trade: float = 0.005
    max_daily_loss: float = 0.02
    max_drawdown: float = 0.1
    max_open_positions: int = 3
    max_leverage: float = 3.0
    max_notional_exposure: float = 1.0
    cooldown_after_loss_streak: int = 3

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def get_risk_config() -> RiskConfig:
    return RiskConfig()
