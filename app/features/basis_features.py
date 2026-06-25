from __future__ import annotations

import pandas as pd


def build_basis_features(frame: pd.DataFrame) -> pd.DataFrame:
    mark = frame["mark_price"].astype(float)
    index = frame["index_price"].astype(float)
    basis_bp = (mark / index - 1.0) * 10_000
    out = pd.DataFrame(index=frame.index)
    out["feature_mark_price"] = mark
    out["feature_index_price"] = index
    out["feature_mark_minus_index_bp"] = basis_bp
    out["feature_basis_zscore"] = (basis_bp - basis_bp.rolling(100).mean()) / basis_bp.rolling(
        100
    ).std()
    out["feature_premium_index"] = frame.get("premium_index", basis_bp / 10_000)
    out["feature_premium_index_change"] = pd.Series(
        out["feature_premium_index"], index=frame.index
    ).diff()
    out["feature_basis_regime"] = (out["feature_basis_zscore"].abs() > 1.5).astype(int)
    out["feature_basis_spike_flag"] = (out["feature_basis_zscore"].abs() > 2.5).astype(int)
    return out
