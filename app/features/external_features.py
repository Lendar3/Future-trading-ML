from __future__ import annotations

import pandas as pd


def build_external_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for prefix in ("es", "nq"):
        close = frame.get(f"{prefix}_close", pd.Series(index=frame.index, dtype=float)).astype(
            float
        )
        out[f"feature_{prefix}_ret_5m"] = close.pct_change(5)
        out[f"feature_{prefix}_ret_15m"] = close.pct_change(15)
        out[f"feature_{prefix}_ret_1h"] = close.pct_change(60)
    out["feature_crypto_beta_to_nq"] = 0.0
    out["feature_btc_nq_corr_20d"] = 0.0
    out["feature_risk_on_flag"] = (out["feature_nq_ret_1h"] > 0).astype(int)
    out["feature_risk_off_flag"] = (out["feature_nq_ret_1h"] < 0).astype(int)
    out["feature_nq_shock_flag"] = (out["feature_nq_ret_15m"].abs() > 0.005).astype(int)
    out["feature_us_session_flag"] = frame.get("feature_us_session_flag", 0)
    return out
