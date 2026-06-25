from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from app.api import routes_models
from app.api.main import app
from app.models.feature_importance import build_feature_importance_summary
from app.models.registry import save_model_artifact
from app.models.train_lightgbm import train_lightgbm_advisor
from tests.unit.test_feature_importance import _make_dataset


def test_feature_importance_endpoint_returns_404_when_missing(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        temp_dir = Path(tmp)

        def _build(model_run_id: str, importance_type: str = "gain", top_n: int = 20):
            return build_feature_importance_summary(
                model_run_id=model_run_id,
                importance_type=importance_type,
                top_n=top_n,
                base_dir=temp_dir,
            )

        monkeypatch.setattr(routes_models, "build_feature_importance_summary", _build)
        client = TestClient(app)
        response = client.get("/models/missing_run/feature-importance")
        assert response.status_code == 404


def test_feature_importance_endpoint_returns_400_on_bad_type(monkeypatch) -> None:
    ds = _make_dataset()
    trained = train_lightgbm_advisor(ds)
    with tempfile.TemporaryDirectory() as tmp:
        temp_dir = Path(tmp)
        save_model_artifact(trained, base_dir=temp_dir)

        def _build(model_run_id: str, importance_type: str = "gain", top_n: int = 20):
            return build_feature_importance_summary(
                model_run_id=model_run_id,
                importance_type=importance_type,
                top_n=top_n,
                base_dir=temp_dir,
            )

        monkeypatch.setattr(routes_models, "build_feature_importance_summary", _build)
        client = TestClient(app)
        response = client.get(
            f"/models/{trained.model_run_id}/feature-importance",
            params={"importance_type": "bad_type"},
        )
        assert response.status_code == 400
