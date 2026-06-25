"""Model artifact registry for the Bybit Futures Edge Lab.

Artifacts are stored on the local filesystem under::

    artifacts/models/{model_run_id}/model.txt      — LightGBM booster text
    artifacts/models/{model_run_id}/metadata.json  — run metadata
    artifacts/models/{model_run_id}/evaluation.json — compact evaluation report

The ``artifacts/`` root is already listed in ``.gitignore``; generated
model files are never committed to version control.

Functions
---------
save_model_artifact(result, base_dir)
    Persist a ``TrainingResult`` (model + metadata) to disk.

load_model_artifact(model_run_id, base_dir)
    Load a ``ModelArtifact`` descriptor pointing at the saved files.
    The LightGBM model can be re-loaded via the returned ``artifact_path``.

load_lgbm_model(model_run_id, base_dir)
    Convenience wrapper: loads and returns the fitted ``lgb.Booster``.

load_model_evaluation(model_run_id, base_dir)
    Load and return parsed ``evaluation.json`` as a dictionary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import lightgbm as lgb

# Avoid circular import: import TrainingResult lazily inside functions.

_DEFAULT_BASE_DIR = Path("artifacts/models")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelArtifact:
    """Metadata descriptor for a persisted model artifact.

    Attributes
    ----------
    model_run_id:
        Unique run identifier (24-char hex).
    artifact_path:
        Absolute path to ``model.txt``.
    metadata_path:
        Absolute path to ``metadata.json``.
    created_at:
        UTC timestamp of when the artifact was saved.
    feature_columns:
        Ordered list of features the model was trained on.
    label_column:
        Name / description of the target variable.
    metrics:
        Training metrics dict.
    params:
        LightGBM hyperparameters used.
    """

    model_run_id: str
    artifact_path: str
    metadata_path: str
    created_at: str  # ISO-8601 UTC string
    evaluation_path: str
    feature_columns: list[str]
    label_column: str
    metrics: dict[str, float]
    params: dict[str, Any]


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_model_artifact(
    result: Any,  # TrainingResult — typed as Any to avoid circular import
    base_dir: Path | str = _DEFAULT_BASE_DIR,
) -> str:
    """Save a fitted LightGBM model and its metadata to *base_dir*.

    Parameters
    ----------
    result:
        A ``TrainingResult`` with a non-None ``.model`` attribute.
    base_dir:
        Root directory for model artifacts.  Defaults to
        ``artifacts/models``.

    Returns
    -------
    str
        Absolute path to the saved ``model.txt`` file.

    Raises
    ------
    ValueError
        If ``result.model`` is ``None`` (no model to save).
    """
    if result.model is None:
        raise ValueError(
            f"TrainingResult for run {result.model_run_id} has no fitted model to save."
        )

    run_dir = Path(base_dir) / result.model_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "model.txt"
    metadata_path = run_dir / "metadata.json"
    evaluation_path = run_dir / "evaluation.json"

    # Save LightGBM booster in text format (portable, version-independent).
    booster: lgb.Booster = result.model.booster_
    booster.save_model(str(model_path))

    # Save metadata.
    now_utc = datetime.now(UTC).isoformat()
    metadata: dict[str, Any] = {
        "model_run_id": result.model_run_id,
        "created_at": now_utc,
        "artifact_path": str(model_path.resolve()),
        "feature_columns": result.feature_columns,
        "label_column": result.label_column,
        "metrics": result.metrics,
        "params": result.best_params,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    evaluation: dict[str, Any] = {
        "model_run_id": result.model_run_id,
        "created_at": now_utc,
        "label_column": result.label_column,
        "feature_count": len(result.feature_columns),
        "row_splits": {
            "rows_total": int(result.metrics.get("rows_total", 0)),
            "rows_train": int(result.metrics.get("rows_train", 0)),
            "rows_valid": int(result.metrics.get("rows_valid", 0)),
            "rows_test": int(result.metrics.get("rows_test", 0)),
        },
        "baseline_comparison": {
            "baseline_majority_accuracy": result.metrics.get("baseline_majority_accuracy"),
            "baseline_random_accuracy": result.metrics.get("baseline_random_accuracy"),
            "baseline_no_edge_accuracy": result.metrics.get("baseline_no_edge_accuracy"),
            "model_vs_majority_accuracy_delta": result.metrics.get(
                "model_vs_majority_accuracy_delta"
            ),
            "model_vs_random_accuracy_delta": result.metrics.get("model_vs_random_accuracy_delta"),
        },
        "class_distribution": {
            "class_count_0": result.metrics.get("class_count_0"),
            "class_count_1": result.metrics.get("class_count_1"),
            "class_count_2": result.metrics.get("class_count_2"),
            "class_ratio_0": result.metrics.get("class_ratio_0"),
            "class_ratio_1": result.metrics.get("class_ratio_1"),
            "class_ratio_2": result.metrics.get("class_ratio_2"),
            "target_class_count": result.metrics.get("target_class_count"),
            "target_max_class_ratio": result.metrics.get("target_max_class_ratio"),
            "target_high_imbalance_flag": result.metrics.get("target_high_imbalance_flag"),
        },
        "metrics": result.metrics,
    }
    evaluation_path.write_text(json.dumps(evaluation, indent=2, default=str), encoding="utf-8")

    return str(model_path.resolve())


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_model_artifact(
    model_run_id: str,
    base_dir: Path | str = _DEFAULT_BASE_DIR,
) -> ModelArtifact:
    """Load the ``ModelArtifact`` descriptor for *model_run_id*.

    Parameters
    ----------
    model_run_id:
        24-char hex run ID.
    base_dir:
        Root directory for model artifacts.

    Returns
    -------
    ModelArtifact

    Raises
    ------
    FileNotFoundError
        If the artifact directory or metadata file does not exist.
    """
    run_dir = Path(base_dir) / model_run_id
    metadata_path = run_dir / "metadata.json"
    model_path = run_dir / "model.txt"
    evaluation_path = run_dir / "evaluation.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No artifact metadata found for model_run_id={model_run_id!r} " f"at {metadata_path}"
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"No model.txt found for model_run_id={model_run_id!r} at {model_path}"
        )

    meta: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))

    return ModelArtifact(
        model_run_id=meta["model_run_id"],
        artifact_path=meta.get("artifact_path", str(model_path.resolve())),
        metadata_path=str(metadata_path.resolve()),
        created_at=meta.get("created_at", ""),
        evaluation_path=str(evaluation_path.resolve()) if evaluation_path.exists() else "",
        feature_columns=meta.get("feature_columns", []),
        label_column=meta.get("label_column", ""),
        metrics=meta.get("metrics", {}),
        params=meta.get("params", {}),
    )


def load_lgbm_model(
    model_run_id: str,
    base_dir: Path | str = _DEFAULT_BASE_DIR,
) -> lgb.Booster:
    """Load and return the fitted ``lgb.Booster`` for *model_run_id*.

    Parameters
    ----------
    model_run_id:
        24-char hex run ID.
    base_dir:
        Root directory for model artifacts.

    Returns
    -------
    lgb.Booster
    """
    artifact = load_model_artifact(model_run_id, base_dir=base_dir)
    return lgb.Booster(model_file=artifact.artifact_path)


def load_model_evaluation(
    model_run_id: str,
    base_dir: Path | str = _DEFAULT_BASE_DIR,
) -> dict[str, Any]:
    """Load and return ``evaluation.json`` for *model_run_id*."""
    eval_path = Path(base_dir) / model_run_id / "evaluation.json"
    if not eval_path.exists():
        raise FileNotFoundError(
            f"No evaluation.json found for model_run_id={model_run_id!r} at {eval_path}"
        )
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    return cast(dict[str, Any], payload)
