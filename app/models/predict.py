from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.models.registry import load_lgbm_model, load_model_artifact


def _prepare_feature_frame(features: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    prepared = pd.DataFrame(index=features.index)
    for col in feature_columns:
        if col in features.columns:
            prepared[col] = pd.to_numeric(features[col], errors="coerce")
        else:
            prepared[col] = 0.0
    return prepared.fillna(0.0).astype(np.float32)


def _normalize_probabilities(
    raw: np.ndarray, rows: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if raw.ndim == 1:
        if raw.shape[0] != rows:
            raise ValueError(
                f"Unsupported prediction shape: expected {rows} rows, got {raw.shape[0]}."
            )
        long_p = np.clip(raw, 0.0, 1.0)
        no_edge_p = 1.0 - long_p
        short_p = np.zeros_like(long_p)
        return no_edge_p, long_p, short_p

    if raw.ndim != 2 or raw.shape[0] != rows:
        raise ValueError(f"Unsupported prediction shape: {raw.shape}.")

    if raw.shape[1] == 3:
        return raw[:, 0], raw[:, 1], raw[:, 2]
    if raw.shape[1] == 2:
        return raw[:, 0], raw[:, 1], np.zeros(rows, dtype=np.float32)
    if raw.shape[1] == 1:
        long_p = np.clip(raw[:, 0], 0.0, 1.0)
        no_edge_p = 1.0 - long_p
        short_p = np.zeros_like(long_p)
        return no_edge_p, long_p, short_p

    raise ValueError(
        "Unsupported prediction shape: " f"{raw.shape}. Expected (n,), (n,1), (n,2), or (n,3)."
    )


def predict_advisor_probabilities(
    features: pd.DataFrame,
    model_run_id: str,
    artifact_base_dir: Path | str = Path("artifacts/models"),
) -> pd.DataFrame:
    artifact = load_model_artifact(model_run_id=model_run_id, base_dir=artifact_base_dir)
    if not artifact.feature_columns:
        raise ValueError(f"Model artifact {model_run_id!r} has no feature_columns in metadata.")

    prepared = _prepare_feature_frame(features=features, feature_columns=artifact.feature_columns)
    booster = load_lgbm_model(model_run_id=model_run_id, base_dir=artifact_base_dir)
    raw = np.asarray(booster.predict(prepared))
    no_edge_p, long_p, short_p = _normalize_probabilities(raw=raw, rows=len(prepared))

    output = pd.DataFrame(index=features.index)
    output["feature_ml_no_edge_probability"] = no_edge_p
    output["feature_ml_long_probability"] = long_p
    output["feature_ml_short_probability"] = short_p
    return output
