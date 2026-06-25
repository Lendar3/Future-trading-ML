from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.models.feature_importance import build_feature_importance_summary
from app.models.registry import load_model_evaluation
from app.pipelines.training_pipeline import run_training_pipeline_from_postgres
from app.reports.readiness_loader import load_readiness_report_payload
from app.storage.database import build_engine, build_session_factory, session_scope
from app.storage.repositories import PostgresModelRepository

router = APIRouter()


@router.post("/train/lightgbm")
def train_lightgbm(
    symbol: str = Query("BTCUSDT"),
    timeframe: str = Query("1m"),
    source: str | None = Query(None),
    feature_set_version: str = Query("v1"),
) -> dict[str, object]:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    try:
        result = run_training_pipeline_from_postgres(
            session_factory=session_factory,
            symbol=symbol,
            timeframe=timeframe,
            source=source,
            feature_set_version=feature_set_version,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"status": "ok", **result.to_dict()}


@router.get("/models/latest")
def latest_model(
    symbol: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
) -> dict[str, list[dict[str, object]]]:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        runs = PostgresModelRepository(session).latest(symbol=symbol, limit=limit)
    return {"model_runs": runs}


@router.get("/models/{model_run_id}/evaluation")
def model_evaluation(model_run_id: str) -> dict[str, object]:
    try:
        report = load_model_evaluation(model_run_id=model_run_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Evaluation artifact not found for model_run_id={model_run_id}.",
        ) from exc

    baseline = report.get("baseline_comparison", {})
    class_distribution = report.get("class_distribution", {})
    row_splits = report.get("row_splits", {})

    return {
        "model_run_id": report.get("model_run_id", model_run_id),
        "metrics": report.get("metrics", {}),
        "baseline_comparison": baseline,
        "class_distribution": class_distribution,
        "rows_train": row_splits.get("rows_train", 0),
        "rows_valid": row_splits.get("rows_valid", 0),
        "rows_test": row_splits.get("rows_test", 0),
        "feature_count": report.get("feature_count", 0),
        "label_column": report.get("label_column", ""),
        "created_at": report.get("created_at", ""),
    }


@router.get("/models/{model_run_id}/feature-importance")
def model_feature_importance(
    model_run_id: str,
    importance_type: str = Query("gain"),
    top_n: int = Query(20, ge=1, le=500),
) -> dict[str, object]:
    try:
        return build_feature_importance_summary(
            model_run_id=model_run_id,
            importance_type=importance_type,
            top_n=top_n,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Feature importance artifact not found for model_run_id={model_run_id}.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/models/{model_run_id}/readiness-report")
def model_readiness_report(
    model_run_id: str,
    importance_type: str = Query("gain"),
    top_n: int = Query(20, ge=1, le=500),
) -> dict[str, object]:
    try:
        return load_readiness_report_payload(
            model_run_id=model_run_id,
            importance_type=importance_type,
            top_n=top_n,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Evaluation artifact not found for model_run_id={model_run_id}.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/models/{model_run_id}/strategy-comparison/latest")
def latest_strategy_comparison_snapshot(model_run_id: str) -> dict[str, object]:
    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        snapshot = PostgresModelRepository(session).latest_strategy_comparison_snapshot(
            model_run_id
        )
    if snapshot is None:
        raise HTTPException(
            status_code=404,
            detail=f"Strategy comparison snapshot not found for model_run_id={model_run_id}.",
        )
    return snapshot
