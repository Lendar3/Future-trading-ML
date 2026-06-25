from __future__ import annotations

import pandas as pd


def apply_funding_filter(signals: pd.DataFrame, max_cost_bp: float = 5.0) -> pd.DataFrame:
    output = signals.copy()
    if "feature_expected_funding_cost" not in output.columns:
        return output
    rejected = output["feature_expected_funding_cost"].abs() * 10_000 > max_cost_bp
    output.loc[rejected, "side"] = "no_trade"
    output.loc[rejected, "risk_rejection_reason"] = "funding cost too high"
    return output
