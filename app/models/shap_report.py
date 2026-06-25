from __future__ import annotations

import pandas as pd


def build_shap_summary(features: pd.DataFrame) -> dict[str, float]:
    feature_cols = [col for col in features.columns if col.startswith("feature_")]
    return {col: 0.0 for col in feature_cols[:25]}
