from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd


def _first_present(row: pd.Series, keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in row and pd.notna(row[key]):
            return row[key]
    return default


def predictions_to_signal_frame(
    dataset: pd.DataFrame,
    probabilities: pd.DataFrame,
    model_run_id: str,
    symbol_fallback: str,
    min_confidence: float = 0.55,
    min_edge_delta: float = 0.05,
    ttl_sec: int = 1800,
    stop_loss_pct: float = 0.005,
    take_profit_pct: float = 0.010,
) -> pd.DataFrame:
    required_prob_cols = {
        "feature_ml_no_edge_probability",
        "feature_ml_long_probability",
        "feature_ml_short_probability",
    }
    missing_prob = required_prob_cols.difference(probabilities.columns)
    if missing_prob:
        raise ValueError(
            f"Prediction frame missing required probability columns: {sorted(missing_prob)}"
        )

    if not dataset.index.equals(probabilities.index):
        probabilities = probabilities.reindex(dataset.index)

    rows: list[dict[str, Any]] = []
    for row_number, (_, row) in enumerate(dataset.iterrows()):
        proba_row = probabilities.iloc[row_number]
        no_edge_p = float(proba_row["feature_ml_no_edge_probability"])
        long_p = float(proba_row["feature_ml_long_probability"])
        short_p = float(proba_row["feature_ml_short_probability"])

        best_edge_prob = max(long_p, short_p)
        edge_delta = best_edge_prob - no_edge_p

        side = "no_trade"
        reason = "below_thresholds"
        if long_p >= short_p and best_edge_prob >= min_confidence and edge_delta >= min_edge_delta:
            side = "long"
            reason = "long_strongest_and_threshold_passed"
        elif short_p > long_p and best_edge_prob >= min_confidence and edge_delta >= min_edge_delta:
            side = "short"
            reason = "short_strongest_and_threshold_passed"

        event_time = pd.Timestamp(_first_present(row, ["event_time"], pd.Timestamp.now(tz="UTC")))
        symbol = str(_first_present(row, ["symbol"], symbol_fallback))
        strategy_name = "advisor_ml"
        confidence = best_edge_prob
        score = edge_delta

        raw_price = _first_present(row, ["close", "feature_close", "entry_ref"], None)
        has_price = raw_price is not None and pd.notna(raw_price) and float(raw_price) > 0
        if has_price:
            entry_ref = float(raw_price)
            if side == "long":
                stop_ref = entry_ref * (1.0 - stop_loss_pct)
                take_ref = entry_ref * (1.0 + take_profit_pct)
            elif side == "short":
                stop_ref = entry_ref * (1.0 + stop_loss_pct)
                take_ref = entry_ref * (1.0 - take_profit_pct)
            else:
                stop_ref = 0.0
                take_ref = 0.0
            price_reason = "price_from_dataset"
        else:
            entry_ref = 0.0
            stop_ref = 0.0
            take_ref = 0.0
            price_reason = "missing_price_reference"

        rationale_json = {
            "model_run_id": model_run_id,
            "probabilities": {
                "no_edge": no_edge_p,
                "long": long_p,
                "short": short_p,
            },
            "thresholds": {
                "min_confidence": min_confidence,
                "min_edge_delta": min_edge_delta,
                "stop_loss_pct": stop_loss_pct,
                "take_profit_pct": take_profit_pct,
                "ttl_sec": ttl_sec,
            },
            "selected_side": side,
            "reason": reason,
            "price_reason": price_reason,
            "feature_set_name": _first_present(row, ["feature_set_name"], None),
            "source": _first_present(row, ["source"], None),
        }
        signal_id_seed = f"{model_run_id}|{symbol}|{event_time.isoformat()}|{side}"
        signal_id = hashlib.sha256(signal_id_seed.encode("utf-8")).hexdigest()[:24]

        rows.append(
            {
                "signal_id": signal_id,
                "symbol": symbol,
                "event_time": event_time,
                "strategy_name": strategy_name,
                "side": side,
                "score": score,
                "confidence": confidence,
                "entry_ref": entry_ref,
                "stop_ref": stop_ref,
                "take_ref": take_ref,
                "ttl_sec": int(ttl_sec),
                "rationale_json": rationale_json,
            }
        )

    return pd.DataFrame(rows, index=dataset.index)


def rank_signals(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return signals
    output = signals.copy()
    output["advisor_rank"] = output["confidence"].rank(ascending=False, method="first")
    return output.sort_values("advisor_rank")
