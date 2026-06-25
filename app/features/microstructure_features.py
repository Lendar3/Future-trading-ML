from __future__ import annotations

import pandas as pd


def build_orderbook_features(orderbook: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=orderbook.index)
    out["feature_best_bid"] = orderbook["best_bid"].astype(float)
    out["feature_best_ask"] = orderbook["best_ask"].astype(float)
    out["feature_mid_price"] = (out["feature_best_bid"] + out["feature_best_ask"]) / 2
    out["feature_spread_bp"] = (out["feature_best_ask"] / out["feature_best_bid"] - 1.0) * 10_000
    out["feature_depth_bid_l1"] = orderbook.get("depth_bid_l1", 0.0)
    out["feature_depth_ask_l1"] = orderbook.get("depth_ask_l1", 0.0)
    denom = out["feature_depth_bid_l1"] + out["feature_depth_ask_l1"]
    out["feature_obi_l1"] = (
        out["feature_depth_bid_l1"] - out["feature_depth_ask_l1"]
    ) / denom.replace(0, pd.NA)
    out["feature_microprice"] = out["feature_mid_price"]
    out["feature_microprice_bias"] = out["feature_microprice"] / out["feature_mid_price"] - 1.0
    out["feature_liquidity_gap_flag"] = (out["feature_spread_bp"] > 10).astype(int)
    return out
