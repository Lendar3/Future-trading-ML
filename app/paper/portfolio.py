from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PaperPortfolio:
    equity: float = 10_000.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    positions: dict[str, float] = field(default_factory=dict)

    @property
    def available_margin(self) -> float:
        return max(self.equity + self.realized_pnl + self.unrealized_pnl, 0.0)

    def apply_fill(self, symbol: str, signed_quantity: float, fee: float) -> None:
        self.positions[symbol] = self.positions.get(symbol, 0.0) + signed_quantity
        self.realized_pnl -= fee
