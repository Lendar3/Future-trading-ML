from __future__ import annotations

import pandas as pd

from app.backtest.costs import CostAssumptions
from app.models.labels import build_edge_labels


def test_cost_model_includes_round_trip_fee_spread_slippage_and_margin() -> None:
    costs = CostAssumptions(
        maker_fee_bp=2,
        taker_fee_bp=5,
        spread_bp=1,
        slippage_bp=2,
        safety_margin_bp=3,
        funding_bp=4,
    )
    assert costs.total_round_trip_bp(taker=True) == 17
    assert costs.min_edge_required_bp(taker=True) == 20


def test_labels_are_net_of_costs_and_prefixed() -> None:
    prices = pd.Series([100.0, 101.0, 104.0, 103.0])
    labels = build_edge_labels(
        prices, horizon=1, costs=CostAssumptions(taker_fee_bp=1, safety_margin_bp=1)
    )
    assert {
        "label_long_edge",
        "label_no_edge",
        "label_short_edge",
        "label_future_net_return",
    }.issubset(labels.columns)
    assert labels["label_future_net_return"].iloc[0] < 0.01
