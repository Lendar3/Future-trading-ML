"""Real LightGBM trainer for the Bybit Futures Edge Lab advisor.

Design
------
* **Chronological split** — data is sorted by ``event_time`` and split at
  fixed percentage boundaries (70 / 15 / 15).  No shuffling is done at any
  point, preserving temporal order and preventing lookahead.
* **Target** — three-class direction label derived from the cost-net label
  columns produced by ``build_edge_labels``:

  * ``0`` → no edge (neither long nor short clears costs)
  * ``1`` → long edge
  * ``2`` → short edge

  Priority order for target column selection:

  1. ``label_class`` (pre-computed 3-class column, if present)
  2. Derived from ``label_long_edge`` and ``label_short_edge`` (mutually
     exclusive by construction in ``build_edge_labels``)
  3. ``label_long_edge`` treated as binary (fallback)

* **NaN handling** — feature NaN ratios are recorded *before* filling so
  metrics capture data quality.  NaNs are then filled with ``0.0``.
* **Deterministic** — fixed ``seed=42`` throughout; no hyperparameter search.
* **Small model** — 100 trees, depth 4, fast to train on any laptop.

No-lookahead guarantee
----------------------
The chronological split is the only temporal boundary enforcement needed here
because the dataset produced by the builder already has ``source_available_time
<= event_time`` for every row.  The split index is computed from row position
(after sorting by ``event_time``), never from the value of the timestamp.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
TRAIN_FRAC = 0.70
VALID_FRAC = 0.15
EARLY_STOPPING_ROUNDS = 20
NAN_MODE_DEFAULT: Literal["zero_fill", "native"] = "zero_fill"
CLASS_WEIGHT_MODE_DEFAULT: Literal["none", "balanced"] = "none"
# test fraction = 1 - TRAIN_FRAC - VALID_FRAC = 0.15

_LGB_PARAMS: dict[str, Any] = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": ["multi_logloss", "multi_error"],
    "num_leaves": 15,
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 10,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingResult:
    """Outcome of a single training run.

    Attributes
    ----------
    model_run_id:
        Deterministic hex ID derived from dataset fingerprint.
    metrics:
        All numeric metrics from the training run.
    best_params:
        Model hyperparameters actually used.
    feature_columns:
        Ordered list of feature column names used for training.
    label_column:
        Name of the target column used.
    model:
        The fitted LightGBM classifier (in-memory).
    artifact_path:
        Filesystem path where the model was saved (empty string if not saved).
    """

    model_run_id: str
    metrics: dict[str, float]
    best_params: dict[str, Any]
    feature_columns: list[str] = field(default_factory=list)
    label_column: str = ""
    model: lgb.LGBMClassifier | None = None
    artifact_path: str = ""


# ---------------------------------------------------------------------------
# Target derivation
# ---------------------------------------------------------------------------


def _derive_target(
    dataset: pd.DataFrame,
    *,
    allow_binary_long_edge_fallback: bool,
) -> tuple[pd.Series, str]:
    """Return (y, label_column_name) for the dataset.

    Preference order:
    1. ``label_class`` — pre-encoded 3-class column.
    2. Derived from ``label_long_edge`` + ``label_short_edge``.
    3. ``label_long_edge`` alone (binary fallback).
    """
    if "label_class" in dataset.columns:
        return dataset["label_class"].copy(), "label_class"

    if "label_long_edge" in dataset.columns and "label_short_edge" in dataset.columns:
        # Encode: 0 = no_edge, 1 = long, 2 = short
        y = pd.Series(0, index=dataset.index, name="label_class_derived", dtype=int)
        y[dataset["label_long_edge"] == 1] = 1
        y[dataset["label_short_edge"] == 1] = 2
        return y, "label_long_edge+label_short_edge→3class"

    # Binary fallback
    if allow_binary_long_edge_fallback and "label_long_edge" in dataset.columns:
        return dataset["label_long_edge"].copy(), "label_long_edge"

    raise ValueError(
        "Unsupported target labels. Expected one of: "
        "'label_class', "
        "'label_long_edge' + 'label_short_edge', "
        "or binary 'label_long_edge' (when explicitly allowed)."
    )


# ---------------------------------------------------------------------------
# Chronological split
# ---------------------------------------------------------------------------


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    valid_frac: float = VALID_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split *df* chronologically into (train, valid, test) subsets.

    The frame is sorted by ``event_time`` before splitting.  No shuffling is
    performed at any stage.

    Parameters
    ----------
    df:
        Dataset with an ``event_time`` column.
    train_frac, valid_frac:
        Fractions for train and validation sets.  Test gets the remainder.

    Returns
    -------
    (train_df, valid_df, test_df)
    """
    if train_frac + valid_frac >= 1.0:
        raise ValueError("train_frac + valid_frac must be < 1.0")
    ordered = df.sort_values("event_time").reset_index(drop=True)
    n = len(ordered)
    i_valid = int(n * train_frac)
    i_test = int(n * (train_frac + valid_frac))
    return (
        ordered.iloc[:i_valid].copy(),
        ordered.iloc[i_valid:i_test].copy(),
        ordered.iloc[i_test:].copy(),
    )


# ---------------------------------------------------------------------------
# NaN auditing
# ---------------------------------------------------------------------------


def _nan_ratios(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, float]:
    """Return per-column NaN ratio for all feature columns that have any NaN."""
    ratios: dict[str, float] = {}
    for col in feature_cols:
        r = float(df[col].isna().mean())
        if r > 0:
            ratios[f"nan_ratio_{col}"] = round(r, 4)
    return ratios


def _apply_nan_mode(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    nan_mode: Literal["zero_fill", "native"],
) -> tuple[pd.DataFrame, list[str]]:
    if nan_mode == "zero_fill":
        df[feature_cols] = df[feature_cols].fillna(0.0)
        return df, feature_cols
    if nan_mode == "native":
        return df, feature_cols
    raise ValueError("nan_mode must be one of: 'zero_fill', 'native'.")


def _baseline_accuracy_metrics(
    *,
    y_train_encoded: np.ndarray,
    y_test_encoded: np.ndarray,
    y_test_raw: pd.Series,
    seed: int,
    test_accuracy: float,
) -> dict[str, float]:
    """Compute simple deterministic baseline accuracies on the test split."""
    # Majority baseline: always predict the most frequent class from train.
    class_counts = np.bincount(y_train_encoded)
    majority_class = int(np.argmax(class_counts))
    majority_pred = np.full_like(y_test_encoded, fill_value=majority_class)
    majority_acc = float(accuracy_score(y_test_encoded, majority_pred))

    # Random-stratified baseline: sample by train class frequencies.
    probs = class_counts / class_counts.sum()
    rng = np.random.default_rng(seed)
    random_pred = rng.choice(np.arange(len(class_counts)), size=len(y_test_encoded), p=probs)
    random_acc = float(accuracy_score(y_test_encoded, random_pred))

    metrics: dict[str, float] = {
        "baseline_majority_accuracy": round(majority_acc, 4),
        "baseline_random_accuracy": round(random_acc, 4),
        "model_vs_majority_accuracy_delta": round(test_accuracy - majority_acc, 4),
        "model_vs_random_accuracy_delta": round(test_accuracy - random_acc, 4),
    }

    # Optional naive "always no_edge" baseline on raw labels (class 0 only).
    if (y_test_raw == 0).any():
        no_edge_pred = np.zeros(len(y_test_raw), dtype=int)
        metrics["baseline_no_edge_accuracy"] = round(
            float(accuracy_score(y_test_raw.to_numpy(dtype=int), no_edge_pred)),
            4,
        )

    return metrics


def _class_distribution_metrics(y_raw: pd.Series) -> dict[str, float]:
    """Return target class counts/ratios and an imbalance flag."""
    y_int = y_raw.to_numpy(dtype=int)
    total = len(y_int)
    metrics: dict[str, float] = {}

    for cls in (0, 1, 2):
        count = int((y_int == cls).sum())
        ratio = float(count / total) if total > 0 else 0.0
        metrics[f"class_count_{cls}"] = float(count)
        metrics[f"class_ratio_{cls}"] = round(ratio, 4)

    target_class_count = int(len(np.unique(y_int)))
    max_ratio = max(metrics["class_ratio_0"], metrics["class_ratio_1"], metrics["class_ratio_2"])
    metrics["target_class_count"] = float(target_class_count)
    metrics["target_max_class_ratio"] = round(float(max_ratio), 4)
    # 1.0 means highly imbalanced warning is active.
    metrics["target_high_imbalance_flag"] = 1.0 if max_ratio >= 0.8 else 0.0
    return metrics


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_lightgbm_advisor(
    dataset: pd.DataFrame,
    *,
    allow_binary_long_edge_fallback: bool = True,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    nan_mode: Literal["zero_fill", "native"] = NAN_MODE_DEFAULT,
    class_weight_mode: Literal["none", "balanced"] = CLASS_WEIGHT_MODE_DEFAULT,
) -> TrainingResult:
    """Train a LightGBM direction-classifier on *dataset*.

    The dataset must be produced by ``build_dataset_from_bars`` or
    ``build_dataset_from_public_tables`` (i.e. contain ``feature_*``,
    ``label_*``, and ``event_time`` columns).

    Parameters
    ----------
    dataset:
        Full feature/label DataFrame.  Rows with missing target are dropped
        before splitting; feature NaNs are filled with ``0.0`` after auditing.

    Returns
    -------
    TrainingResult
        Contains the fitted model, metrics, params, and feature list.
    """
    feature_cols = sorted(
        c
        for c in dataset.columns
        if c.startswith("feature_")
        and c != "feature_set_name"
        and pd.api.types.is_numeric_dtype(dataset[c])
    )
    if not feature_cols:
        raise ValueError("Dataset has no feature_* columns.")
    label_cols = [c for c in dataset.columns if c.startswith("label_")]
    if not label_cols:
        raise ValueError("Dataset has no label_* columns.")

    # ---- Derive target -------------------------------------------------------
    y_all, label_column = _derive_target(
        dataset,
        allow_binary_long_edge_fallback=allow_binary_long_edge_fallback,
    )
    valid_mask = y_all.notna()
    df = dataset.loc[valid_mask].copy()
    y_all = y_all.loc[valid_mask]

    rows_total = len(df)
    if rows_total < 30:
        raise ValueError(f"Too few rows after dropping missing labels ({rows_total} < 30).")

    # ---- Record NaN ratios BEFORE filling ------------------------------------
    nan_audit = _nan_ratios(df, feature_cols)

    # ---- Apply NaN mode ------------------------------------------------------
    df, feature_cols = _apply_nan_mode(df, feature_cols, nan_mode=nan_mode)
    y_all = y_all.loc[df.index]

    # ---- Chronological split -------------------------------------------------
    df["__y__"] = y_all.values
    train_df, valid_df, test_df = chronological_split(df)

    X_train = train_df[feature_cols].astype(np.float32)
    X_valid = valid_df[feature_cols].astype(np.float32)
    X_test = test_df[feature_cols].astype(np.float32)

    # Encode labels to 0-based integers.
    le = LabelEncoder()
    le.fit(df["__y__"])
    y_train = le.transform(train_df["__y__"])
    y_valid = le.transform(valid_df["__y__"])
    y_test = le.transform(test_df["__y__"])
    n_classes = len(le.classes_)
    if n_classes < 2:
        raise ValueError("Target must contain at least 2 classes after preprocessing.")

    # ---- Build LightGBM params + trainer config ------------------------------
    training_config: dict[str, Any] = {
        "allow_binary_long_edge_fallback": allow_binary_long_edge_fallback,
        "early_stopping_rounds": early_stopping_rounds,
        "nan_mode": nan_mode,
        "class_weight_mode": class_weight_mode,
    }
    lgbm_params = dict(_LGB_PARAMS)
    lgbm_params["num_class"] = n_classes
    if n_classes == 2:
        lgbm_params["objective"] = "binary"
        lgbm_params["metric"] = ["binary_logloss", "binary_error"]
        del lgbm_params["num_class"]
    if class_weight_mode == "balanced":
        lgbm_params["class_weight"] = "balanced"
    elif class_weight_mode != "none":
        raise ValueError("class_weight_mode must be one of: 'none', 'balanced'.")
    if early_stopping_rounds < 0:
        raise ValueError("early_stopping_rounds must be >= 0.")

    # ---- Fit -----------------------------------------------------------------
    model = lgb.LGBMClassifier(**{k: v for k, v in lgbm_params.items() if k != "metric"})
    callbacks: list[Callable[..., Any]] = [lgb.log_evaluation(period=-1)]
    if early_stopping_rounds > 0:
        callbacks.append(
            lgb.early_stopping(
                stopping_rounds=early_stopping_rounds,
                verbose=False,
            )
        )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=callbacks,
    )

    # ---- Evaluate ------------------------------------------------------------
    def _acc(X: pd.DataFrame, y: np.ndarray) -> float:
        return float(accuracy_score(y, model.predict(X)))

    def _logloss(X: pd.DataFrame, y: np.ndarray) -> float:
        proba = model.predict_proba(X)
        # sklearn log_loss expects labels= for multi-class.
        return float(log_loss(y, proba, labels=list(range(n_classes))))

    train_acc = _acc(X_train, y_train)
    valid_acc = _acc(X_valid, y_valid)
    test_acc = _acc(X_test, y_test)
    valid_ll = _logloss(X_valid, y_valid)
    test_ll = _logloss(X_test, y_test)
    class_dist = _class_distribution_metrics(df["__y__"])
    baselines = _baseline_accuracy_metrics(
        y_train_encoded=y_train,
        y_test_encoded=y_test,
        y_test_raw=test_df["__y__"],
        seed=RANDOM_SEED,
        test_accuracy=test_acc,
    )

    # ---- Build metrics dict --------------------------------------------------
    metrics: dict[str, float] = {
        "rows_total": float(rows_total),
        "rows_train": float(len(train_df)),
        "rows_valid": float(len(valid_df)),
        "rows_test": float(len(test_df)),
        "feature_count": float(len(feature_cols)),
        # Backward-compatible aliases used by existing API/dashboard/tests.
        "rows": float(rows_total),
        "features": float(len(feature_cols)),
        "n_classes": float(n_classes),
        "train_accuracy": round(train_acc, 4),
        "valid_accuracy": round(valid_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "valid_log_loss": round(valid_ll, 4),
        "test_log_loss": round(test_ll, 4),
        "best_iteration": float(
            model.best_iteration_ if getattr(model, "best_iteration_", None) else model.n_estimators
        ),
        "early_stopping_enabled": 1.0 if early_stopping_rounds > 0 else 0.0,
    }
    metrics["nan_mode"] = 0.0 if nan_mode == "zero_fill" else 1.0
    metrics["class_weight_mode"] = 0.0 if class_weight_mode == "none" else 1.0
    metrics.update(nan_audit)
    metrics.update(class_dist)
    metrics.update(baselines)

    # ---- Deterministic run ID ------------------------------------------------
    run_fingerprint = (
        f"{rows_total}-{len(feature_cols)}-{label_column}" f"-{dataset['event_time'].iloc[-1]}"
    )
    model_run_id = hashlib.sha256(run_fingerprint.encode()).hexdigest()[:24]

    return TrainingResult(
        model_run_id=model_run_id,
        metrics=metrics,
        best_params={
            "lgbm_params": {k: v for k, v in lgbm_params.items()},
            "training_config": {k: v for k, v in training_config.items()},
            # Backward-compatible flat aliases.
            **{k: v for k, v in lgbm_params.items()},
            **{k: v for k, v in training_config.items()},
        },
        feature_columns=feature_cols,
        label_column=label_column,
        model=model,
        artifact_path="",
    )


if __name__ == "__main__":
    print("LightGBM real trainer. " "Build a dataset and call train_lightgbm_advisor().")
