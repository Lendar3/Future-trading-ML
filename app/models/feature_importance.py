from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.models.registry import load_lgbm_model, load_model_artifact

_ALLOWED_IMPORTANCE_TYPES = {"gain", "split"}


def build_feature_importance_summary(
    model_run_id: str,
    *,
    importance_type: str = "gain",
    top_n: int = 20,
    base_dir: Path | str = Path("artifacts/models"),
) -> dict[str, Any]:
    if importance_type not in _ALLOWED_IMPORTANCE_TYPES:
        allowed = ", ".join(sorted(_ALLOWED_IMPORTANCE_TYPES))
        raise ValueError(f"Unsupported importance_type={importance_type!r}. Allowed: {allowed}.")
    if top_n < 1:
        raise ValueError("top_n must be >= 1.")

    artifact = load_model_artifact(model_run_id=model_run_id, base_dir=base_dir)
    booster = load_lgbm_model(model_run_id=model_run_id, base_dir=base_dir)

    metadata_features = list(artifact.feature_columns)
    booster_features = list(booster.feature_name())
    raw_importance = list(booster.feature_importance(importance_type=importance_type))
    importance_by_feature = {
        feature: float(value)
        for feature, value in zip(booster_features, raw_importance, strict=True)
    }

    warning: str | None = None
    if len(metadata_features) != len(booster_features) or (
        set(metadata_features) != set(booster_features)
    ):
        warning = (
            "Feature list mismatch between metadata and booster; "
            "importance summary aligns by feature name."
        )

    ordered_features = metadata_features if metadata_features else booster_features
    all_rows: list[dict[str, Any]] = []
    for feature in ordered_features:
        all_rows.append(
            {
                "feature": feature,
                "importance": float(importance_by_feature.get(feature, 0.0)),
            }
        )

    all_rows = sorted(all_rows, key=lambda row: float(row["importance"]), reverse=True)
    total_importance = float(sum(float(row["importance"]) for row in all_rows))
    for index, row in enumerate(all_rows, start=1):
        row["rank"] = index
        row["importance_share"] = (
            float(float(row["importance"]) / total_importance) if total_importance > 0 else 0.0
        )

    payload: dict[str, Any] = {
        "model_run_id": model_run_id,
        "importance_type": importance_type,
        "feature_count": len(all_rows),
        "top_features": all_rows[:top_n],
        "all_features": all_rows,
        "created_at": datetime.now(UTC).isoformat(),
    }
    if warning:
        payload["warning"] = warning
    return payload
