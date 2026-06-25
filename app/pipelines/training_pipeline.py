"""Training pipeline: bars → features → labels → LightGBM model run.

Feature-set modes
-----------------
* ``feature_set_version="v1"`` (default) — legacy OHLCV path.
* ``feature_set_version="v1.6_public"`` — enriched path that also fetches
  and joins mark/index/funding/OI/long-short tables from PostgreSQL.

Artifact saving
---------------
After training, the fitted model is saved to::

    artifacts/models/{model_run_id}/model.txt
    artifacts/models/{model_run_id}/metadata.json

The artifact path is recorded in ``params_json`` so it can be retrieved
from the ``model_runs`` table without a schema migration.
``save_artifact=False`` skips the save (used in tests and smoke checks).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sqlalchemy.orm import sessionmaker

from app.core.clock import utc_now
from app.features.dataset_builder import (
    FEATURE_SET_PUBLIC,
    build_dataset_from_bars,
    build_dataset_from_public_tables,
)
from app.models.registry import save_model_artifact
from app.models.train_lightgbm import train_lightgbm_advisor
from app.storage.database import session_scope
from app.storage.repositories import (
    PostgresMarketDataRepository,
    PostgresModelRepository,
    PostgresPublicMarketRepository,
)

_DEFAULT_ARTIFACT_BASE = Path("artifacts/models")


@dataclass(frozen=True)
class TrainingPipelineResult:
    model_run_id: str
    symbol: str
    timeframe: str
    dataset_rows: int
    metrics: dict[str, float]
    best_params: dict[str, object]
    artifact_path: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "model_run_id": self.model_run_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "dataset_rows": self.dataset_rows,
            "metrics": self.metrics,
            "best_params": self.best_params,
            "artifact_path": self.artifact_path,
        }


def run_training_pipeline_from_postgres(
    session_factory: sessionmaker,
    symbol: str,
    timeframe: str,
    source: str | None = None,
    feature_set_version: str = "v1",
    label_horizon: int = 3,
    allow_binary_long_edge_fallback: bool = True,
    early_stopping_rounds: int = 20,
    nan_mode: Literal["zero_fill", "native"] = "zero_fill",
    class_weight_mode: Literal["none", "balanced"] = "none",
    save_artifact: bool = True,
    artifact_base_dir: Path | str = _DEFAULT_ARTIFACT_BASE,
) -> TrainingPipelineResult:
    """Run the full training pipeline and persist results.

    Parameters
    ----------
    session_factory:
        SQLAlchemy session factory backed by PostgreSQL.
    symbol:
        Instrument, e.g. ``"BTCUSDT"``.
    timeframe:
        Bar interval, e.g. ``"1"`` (Bybit 1-minute).
    source:
        Optional source filter for ``raw_bars`` and sparse public tables.
    feature_set_version:
        ``"v1"`` or ``"v1.6_public"``.
    label_horizon:
        Bars-ahead for net-of-cost label construction.
    save_artifact:
        If ``True`` (default), save the fitted model to disk.  Set to
        ``False`` in unit/smoke tests to skip filesystem writes.
    artifact_base_dir:
        Root directory for model artifacts.
    """
    enriched = feature_set_version == FEATURE_SET_PUBLIC

    # ---- Fetch data ----------------------------------------------------------
    with session_scope(session_factory) as session:
        market_repo = PostgresMarketDataRepository(session)
        bars = market_repo.fetch_raw_bars(symbol=symbol, timeframe=timeframe, source=source)

        if enriched:
            public_repo = PostgresPublicMarketRepository(session)
            mark_bars = public_repo.fetch_mark_bars(symbol=symbol, timeframe=timeframe)
            index_bars = public_repo.fetch_index_bars(symbol=symbol, timeframe=timeframe)
            funding = public_repo.fetch_funding(symbol=symbol, source=source)
            open_interest = public_repo.fetch_open_interest(symbol=symbol, source=source)
            long_short = public_repo.fetch_long_short_ratio(symbol=symbol, source=source)

    if bars.empty:
        raise ValueError(f"No raw bars found for {symbol} {timeframe}.")

    # ---- Build dataset -------------------------------------------------------
    if enriched:
        dataset = build_dataset_from_public_tables(
            bars=bars,
            symbol=symbol,
            timeframe=timeframe,
            mark_bars=mark_bars if not mark_bars.empty else None,
            index_bars=index_bars if not index_bars.empty else None,
            funding=funding if not funding.empty else None,
            open_interest=open_interest if not open_interest.empty else None,
            long_short_ratio=long_short if not long_short.empty else None,
            feature_set_name=feature_set_version,
            label_horizon=label_horizon,
        )
    else:
        dataset = build_dataset_from_bars(
            bars=bars,
            symbol=symbol,
            timeframe=timeframe,
            feature_set_name=feature_set_version,
            label_horizon=label_horizon,
        )

    # ---- Pre-processing (drop unlabelled horizon rows) ----------------------
    # train_lightgbm_advisor handles NaN drops and feature fills internally,
    # but we drop the guaranteed-NaN tail (label horizon) here so dataset_rows
    # reflects rows that actually enter training.
    label_columns = [c for c in dataset.columns if c.startswith("label_")]
    dataset = dataset.dropna(subset=label_columns).copy()
    if dataset.empty:
        raise ValueError("Dataset is empty after dropping rows with unavailable labels.")

    # ---- Train ---------------------------------------------------------------
    result = train_lightgbm_advisor(
        dataset,
        allow_binary_long_edge_fallback=allow_binary_long_edge_fallback,
        early_stopping_rounds=early_stopping_rounds,
        nan_mode=nan_mode,
        class_weight_mode=class_weight_mode,
    )

    # ---- Save artifact -------------------------------------------------------
    artifact_path = ""
    if save_artifact and result.model is not None:
        artifact_path = save_model_artifact(result, base_dir=artifact_base_dir)

    # ---- Persist model run to PostgreSQL ------------------------------------
    params_to_store = dict(result.best_params)
    if artifact_path:
        params_to_store["artifact_path"] = artifact_path

    with session_scope(session_factory) as session:
        repo = PostgresModelRepository(session)
        repo.upsert_model_run(
            model_run_id=result.model_run_id,
            symbol=symbol,
            created_at=utc_now(),
            feature_set_version=feature_set_version,
            metrics_json=result.metrics,
            params_json=params_to_store,
        )

    return TrainingPipelineResult(
        model_run_id=result.model_run_id,
        symbol=symbol,
        timeframe=timeframe,
        dataset_rows=len(dataset),
        metrics=result.metrics,
        best_params=result.best_params,
        artifact_path=artifact_path,
    )
