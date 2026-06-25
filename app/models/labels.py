from __future__ import annotations

import pandas as pd

from app.backtest.costs import CostAssumptions, bp_to_return


def build_edge_labels(
    prices: pd.Series,
    horizon: int,
    costs: CostAssumptions | None = None,
) -> pd.DataFrame:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    cost_model = costs or CostAssumptions()
    future_return = prices.shift(-horizon) / prices - 1.0
    threshold = bp_to_return(cost_model.min_edge_required_bp(taker=True))

    labels = pd.DataFrame(index=prices.index)
    labels["label_future_net_return"] = future_return - bp_to_return(
        cost_model.total_round_trip_bp(taker=True)
    )
    labels["label_long_edge"] = (labels["label_future_net_return"] > threshold).astype(int)
    labels["label_short_edge"] = (labels["label_future_net_return"] < -threshold).astype(int)
    labels["label_no_edge"] = (
        (labels["label_long_edge"] == 0) & (labels["label_short_edge"] == 0)
    ).astype(int)
    return labels
