from __future__ import annotations

from app.models.feature_importance import build_feature_importance_summary
from app.models.registry import load_model_evaluation
from app.reports.readiness import build_research_readiness_report
from app.storage.database import build_engine, build_session_factory, session_scope
from app.storage.repositories import PostgresModelRepository


def load_readiness_report_payload(
    model_run_id: str,
    *,
    importance_type: str = "gain",
    top_n: int = 20,
) -> dict[str, object]:
    evaluation = load_model_evaluation(model_run_id=model_run_id)
    extra_warnings: list[str] = []
    feature_importance: dict[str, object] | None = None
    try:
        feature_importance = build_feature_importance_summary(
            model_run_id=model_run_id,
            importance_type=importance_type,
            top_n=top_n,
        )
    except FileNotFoundError:
        extra_warnings.append("Feature importance artifact is unavailable for this model run.")

    strategy_comparison_payload: dict[str, object] | None = None
    engine = build_engine()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        snapshot = PostgresModelRepository(session).latest_strategy_comparison_snapshot(
            model_run_id
        )
    if snapshot is not None:
        strategy_comparison_payload = snapshot.get("comparison_json")

    return build_research_readiness_report(
        model_run_id=model_run_id,
        model_evaluation_payload=evaluation,
        feature_importance_payload=feature_importance,
        strategy_comparison_payload=strategy_comparison_payload,
        extra_warnings=extra_warnings,
    )
