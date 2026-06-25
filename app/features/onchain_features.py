from __future__ import annotations

import pandas as pd


def build_onchain_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for name in (
        "btc_exchange_netflow",
        "eth_exchange_netflow",
        "stablecoin_exchange_netflow",
        "stablecoin_supply_change",
        "large_wallet_inflow",
        "large_wallet_outflow",
        "exchange_reserve_change",
        "miner_flow",
        "whale_activity_flag",
    ):
        out[f"feature_{name}"] = frame.get(name, 0.0)
    return out
