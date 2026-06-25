from __future__ import annotations

from contextlib import contextmanager

from fastapi.testclient import TestClient

from app.api import routes_models
from app.api.main import app
from app.reports import readiness_loader


def test_readiness_report_endpoint_success(monkeypatch) -> None:
    model_run_id = "readiness_ok_001"

    def _load(model_run_id: str) -> dict[str, object]:
        return {
            "model_run_id": model_run_id,
            "rows_test": 20,
            "metrics": {"test_accuracy": 0.61},
            "baseline_comparison": {
                "baseline_majority_accuracy": 0.52,
                "baseline_random_accuracy": 0.44,
            },
            "class_distribution": {
                "target_max_class_ratio": 0.64,
                "target_high_imbalance_flag": 0.0,
            },
        }

    def _importance(model_run_id: str, importance_type: str = "gain", top_n: int = 20):
        return {
            "model_run_id": model_run_id,
            "feature_count": 8,
            "top_features": [
                {
                    "feature": "feature_a",
                    "importance": 5.0,
                    "rank": 1,
                    "importance_share": 0.30,
                }
            ],
        }

    class _Repo:
        def __init__(self, session) -> None:
            self.session = session

        def latest_strategy_comparison_snapshot(self, model_run_id: str):
            return {
                "snapshot_id": "scs_1",
                "model_run_id": model_run_id,
                "symbol": "BTCUSDT",
                "timeframe": "1m",
                "feature_set_version": "v1",
                "created_at": "2026-01-01T00:00:00+00:00",
                "comparison_json": {
                    "quality_gate_passed": True,
                    "quality_gates": {
                        "ml_has_trades": True,
                        "ml_beats_best_rule_total_return": True,
                        "ml_beats_best_rule_profit_factor": True,
                        "enough_folds": True,
                        "positive_fold_ratio_ok": True,
                    },
                    "quality_gate_warnings": [],
                    "summary": {"best_strategy_by_total_return": "ml_advisor"},
                },
            }

    @contextmanager
    def _session_scope(_session_factory):
        yield object()

    monkeypatch.setattr(readiness_loader, "load_model_evaluation", _load)
    monkeypatch.setattr(readiness_loader, "build_feature_importance_summary", _importance)
    monkeypatch.setattr(readiness_loader, "build_engine", lambda: object())
    monkeypatch.setattr(readiness_loader, "build_session_factory", lambda _engine: object())
    monkeypatch.setattr(readiness_loader, "session_scope", _session_scope)
    monkeypatch.setattr(readiness_loader, "PostgresModelRepository", _Repo)

    client = TestClient(app)
    response = client.get(f"/models/{model_run_id}/readiness-report")
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_run_id"] == model_run_id
    assert "readiness_status" in payload
    assert "readiness_score" in payload
    assert "readiness_checks" in payload
    assert payload["strategy_comparison_summary"]["available"] is True
    assert payload["readiness_status"] == "paper_candidate"


def test_readiness_report_endpoint_returns_404_when_evaluation_missing(monkeypatch) -> None:
    def _load(model_run_id: str) -> dict[str, object]:
        raise FileNotFoundError("missing evaluation")

    monkeypatch.setattr(readiness_loader, "load_model_evaluation", _load)
    client = TestClient(app)
    response = client.get("/models/missing_eval_001/readiness-report")
    assert response.status_code == 404


def test_readiness_report_remains_conservative_when_snapshot_missing(monkeypatch) -> None:
    model_run_id = "readiness_missing_snapshot_001"

    def _load(model_run_id: str) -> dict[str, object]:
        return {
            "model_run_id": model_run_id,
            "rows_test": 20,
            "metrics": {"test_accuracy": 0.61},
            "baseline_comparison": {
                "baseline_majority_accuracy": 0.52,
                "baseline_random_accuracy": 0.44,
            },
            "class_distribution": {
                "target_max_class_ratio": 0.64,
                "target_high_imbalance_flag": 0.0,
            },
        }

    def _importance(model_run_id: str, importance_type: str = "gain", top_n: int = 20):
        return {
            "model_run_id": model_run_id,
            "feature_count": 8,
            "top_features": [
                {
                    "feature": "feature_a",
                    "importance": 5.0,
                    "rank": 1,
                    "importance_share": 0.30,
                }
            ],
        }

    class _Repo:
        def __init__(self, session) -> None:
            self.session = session

        def latest_strategy_comparison_snapshot(self, model_run_id: str):
            return None

    @contextmanager
    def _session_scope(_session_factory):
        yield object()

    monkeypatch.setattr(readiness_loader, "load_model_evaluation", _load)
    monkeypatch.setattr(readiness_loader, "build_feature_importance_summary", _importance)
    monkeypatch.setattr(readiness_loader, "build_engine", lambda: object())
    monkeypatch.setattr(readiness_loader, "build_session_factory", lambda _engine: object())
    monkeypatch.setattr(readiness_loader, "session_scope", _session_scope)
    monkeypatch.setattr(readiness_loader, "PostgresModelRepository", _Repo)

    client = TestClient(app)
    response = client.get(f"/models/{model_run_id}/readiness-report")
    assert response.status_code == 200
    payload = response.json()
    assert payload["strategy_comparison_summary"]["available"] is False
    assert payload["readiness_status"] == "research_only"
    assert any("Strategy comparison data is missing." in r for r in payload["blocking_reasons"])


def test_latest_strategy_comparison_snapshot_endpoint(monkeypatch) -> None:
    model_run_id = "model_snapshot_api_001"

    class _Repo:
        def __init__(self, session) -> None:
            self.session = session

        def latest_strategy_comparison_snapshot(self, model_run_id: str):
            return {
                "snapshot_id": "scs_test_api",
                "model_run_id": model_run_id,
                "symbol": "BTCUSDT",
                "timeframe": "1m",
                "feature_set_version": "v1",
                "created_at": "2026-01-01T00:00:00+00:00",
                "comparison_json": {
                    "model_run_id": model_run_id,
                    "symbol": "BTCUSDT",
                    "feature_set_version": "v1",
                    "compared_strategies": ["ml_advisor", "mean_reversion"],
                    "comparison_rows": [],
                    "summary": {},
                    "quality_gates": {},
                    "quality_gate_passed": False,
                    "quality_gate_warnings": [],
                    "comparison_config": {},
                },
            }

    @contextmanager
    def _session_scope(_session_factory):
        yield object()

    monkeypatch.setattr(routes_models, "build_engine", lambda: object())
    monkeypatch.setattr(routes_models, "build_session_factory", lambda _engine: object())
    monkeypatch.setattr(routes_models, "session_scope", _session_scope)
    monkeypatch.setattr(routes_models, "PostgresModelRepository", _Repo)

    client = TestClient(app)
    response = client.get(f"/models/{model_run_id}/strategy-comparison/latest")
    assert response.status_code == 200
    payload = response.json()
    assert payload["snapshot_id"] == "scs_test_api"
    assert payload["model_run_id"] == model_run_id


def test_latest_strategy_comparison_snapshot_endpoint_returns_404_when_missing(monkeypatch) -> None:
    class _Repo:
        def __init__(self, session) -> None:
            self.session = session

        def latest_strategy_comparison_snapshot(self, model_run_id: str):
            return None

    @contextmanager
    def _session_scope(_session_factory):
        yield object()

    monkeypatch.setattr(routes_models, "build_engine", lambda: object())
    monkeypatch.setattr(routes_models, "build_session_factory", lambda _engine: object())
    monkeypatch.setattr(routes_models, "session_scope", _session_scope)
    monkeypatch.setattr(routes_models, "PostgresModelRepository", _Repo)

    client = TestClient(app)
    response = client.get("/models/missing_001/strategy-comparison/latest")
    assert response.status_code == 404
