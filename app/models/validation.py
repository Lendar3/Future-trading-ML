from __future__ import annotations

import pandas as pd


def validate_feature_timestamps(frame: pd.DataFrame) -> None:
    if "event_time" not in frame.columns:
        raise ValueError("feature frame must include event_time")
    if "source_available_time" in frame.columns:
        leaked = frame["source_available_time"] > frame["event_time"]
        if leaked.any():
            raise ValueError("feature frame contains rows unavailable at event_time")


def assert_no_random_cv(cv_name: str) -> None:
    if "shuffle" in cv_name.lower() or "random" in cv_name.lower():
        raise ValueError("Random shuffled cross-validation is forbidden for time series.")
