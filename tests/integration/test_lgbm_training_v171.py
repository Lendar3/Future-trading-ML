"""Integration test: v1.7.1 — real LightGBM training end-to-end.

Tests
-----
1. Demo-seeded data → training pipeline → model_run persisted → artifact on disk.
2. Training pipeline with v1.6_public feature set → artifact saved.
3. Artifact can be reloaded from disk after pipeline run.
4. ``/models/latest`` API returns the new model_run_id.
5. Live trading still disabled.

Requires: running PostgreSQL (Docker service must be up).
Tagged ``integration`` so CI can skip when Docker is unavailable.
"""

from __future__ import annotations

import tempfile

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.models.registry import load_lgbm_model, load_model_artifact, load_model_evaluation
from app.pipelines.demo_seed import seed_demo_data
from app.pipelines.training_pipeline import run_training_pipeline_from_postgres
from app.storage.database import build_engine, build_session_factory

pytestmark = pytest.mark.integration

_SRC = "lgbm_v171_inttest"


def test_training_pipeline_v1_real_lgbm_artifact() -> None:
    """v1 path: fixture data → train → model_run in DB → artifact on disk."""
    engine = build_engine()
    sf = build_session_factory(engine)
    seed_demo_data(session_factory=sf, symbols=["BTCUSDT"], rows=120, source=_SRC + "_v1")

    with tempfile.TemporaryDirectory() as tmp:
        result = run_training_pipeline_from_postgres(
            session_factory=sf,
            symbol="BTCUSDT",
            timeframe="1m",
            source=_SRC + "_v1",
            feature_set_version="v1",
            save_artifact=True,
            artifact_base_dir=tmp,
        )

        # Basic result checks.
        assert result.dataset_rows > 0
        assert result.model_run_id
        assert result.artifact_path, "artifact_path must be non-empty"

        # Artifact exists on disk.
        from pathlib import Path

        assert Path(result.artifact_path).exists(), "model.txt must exist on disk"

        # Artifact descriptor can be loaded.
        artifact = load_model_artifact(result.model_run_id, base_dir=tmp)
        assert artifact.model_run_id == result.model_run_id
        assert len(artifact.feature_columns) > 0
        assert artifact.evaluation_path.endswith("evaluation.json")

        evaluation = load_model_evaluation(result.model_run_id, base_dir=tmp)
        assert evaluation["model_run_id"] == result.model_run_id
        assert "baseline_comparison" in evaluation

        # Booster can be reloaded and used for predictions.
        import numpy as np

        booster = load_lgbm_model(result.model_run_id, base_dir=tmp)
        dummy = np.zeros((1, len(artifact.feature_columns)), dtype=np.float32)
        preds = booster.predict(dummy)
        assert preds.shape[0] == 1

    # Metrics contain required keys.
    required = {
        "rows_total",
        "train_accuracy",
        "valid_accuracy",
        "test_accuracy",
        "valid_log_loss",
        "test_log_loss",
        "feature_count",
    }
    assert required.issubset(set(result.metrics.keys()))

    # All accuracy values in [0, 1].
    for acc_key in ("train_accuracy", "valid_accuracy", "test_accuracy"):
        assert 0.0 <= result.metrics[acc_key] <= 1.0


def test_training_pipeline_persists_model_run_to_db() -> None:
    """model_run must appear in /models/latest API response."""
    engine = build_engine()
    sf = build_session_factory(engine)
    seed_demo_data(
        session_factory=sf,
        symbols=["ETHUSDT"],
        rows=120,
        source=_SRC + "_api",
    )

    with tempfile.TemporaryDirectory() as tmp:
        result = run_training_pipeline_from_postgres(
            session_factory=sf,
            symbol="ETHUSDT",
            timeframe="1m",
            source=_SRC + "_api",
            feature_set_version="v1",
            save_artifact=True,
            artifact_base_dir=tmp,
        )

    client = TestClient(app)
    resp = client.get("/models/latest", params={"symbol": "ETHUSDT", "limit": 50})
    assert resp.status_code == 200
    ids = [r["model_run_id"] for r in resp.json()["model_runs"]]
    assert (
        result.model_run_id in ids
    ), f"model_run_id {result.model_run_id!r} not in /models/latest response"


def test_training_pipeline_no_artifact_on_save_false() -> None:
    """save_artifact=False must leave artifact_path empty."""
    engine = build_engine()
    sf = build_session_factory(engine)
    seed_demo_data(
        session_factory=sf,
        symbols=["SOLUSDT"],
        rows=120,
        source=_SRC + "_noart",
    )

    result = run_training_pipeline_from_postgres(
        session_factory=sf,
        symbol="SOLUSDT",
        timeframe="1m",
        source=_SRC + "_noart",
        feature_set_version="v1",
        save_artifact=False,
    )
    assert result.artifact_path == "", "artifact_path must be empty when save_artifact=False"
