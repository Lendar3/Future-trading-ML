from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlippageModel:
    base_slippage_bp: float = 2.0
    depth_penalty_bp: float = 0.0

    def estimate_bp(self, spread_bp: float, notional: float, depth_notional: float | None) -> float:
        if depth_notional is None or depth_notional <= 0:
            return self.base_slippage_bp + spread_bp
        depth_ratio = max(notional / depth_notional, 0.0)
        return self.base_slippage_bp + spread_bp + self.depth_penalty_bp * depth_ratio
