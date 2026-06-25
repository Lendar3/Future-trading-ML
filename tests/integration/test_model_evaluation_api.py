from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from app.api import routes_models
from app.api.main import app
from app.models.registry import load_model_evaluation


def test_model_evaluation_endpoint_returns_report(monkeypatch) -> None:
    model_run_id = "eval_api_ok_001"
    evaluation = {
        "model_run_id": model_run_id,
        "created_at": "2026-05-10T00:00:00+00:00",
        "label_column": "label_class",
        "feature_count": 42,
        "row_splits": {"rows_total": 100, "rows_train": 70, "rows_valid": 15, "rows_test": 15},
        "baseline_comparison": {
            "baseline_majority_accuracy": 0.5,
            "baseline_random_accuracy": 0.4,
            "model_vs_majority_accuracy_delta": 0.1,
            "model_vs_random_accuracy_delta": 0.2,
        },
        "class_distribution": {
            "target_class_count": 3,
            "target_max_class_ratio": 0.7,
            "target_high_imbalance_flag": 0.0,
        },
        "metrics": {"test_accuracy": 0.6},
    }

    with tempfile.TemporaryDirectory() as tmp:
        temp_dir = Path(tmp)
        run_dir = temp_dir / model_run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "evaluation.json").write_text(json.dumps(evaluation), encoding="utf-8")

        def _load(model_run_id: str) -> dict[str, object]:
            return load_model_evaluation(model_run_id, base_dir=temp_dir)

        monkeypatch.setattr(routes_models, "load_model_evaluation", _load)
        client = TestClient(app)
        response = client.get(f"/models/{model_run_id}/evaluation")
        assert response.status_code == 200
        payload = response.json()
        assert payload["model_run_id"] == model_run_id
        assert payload["rows_train"] == 70
        assert payload["rows_valid"] == 15
        assert payload["rows_test"] == 15
        assert payload["feature_count"] == 42
        assert "baseline_comparison" in payload
        assert "class_distribution" in payload


def test_model_evaluation_endpoint_returns_404_when_missing(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        temp_dir = Path(tmp)

        def _load(model_run_id: str) -> dict[str, object]:
            return load_model_evaluation(model_run_id, base_dir=temp_dir)

        monkeypatch.setattr(routes_models, "load_model_evaluation", _load)
        client = TestClient(app)
        response = client.get("/models/does_not_exist/evaluation")
        assert response.status_code == 404
