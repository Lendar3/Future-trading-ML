from __future__ import annotations

import pandas as pd
import streamlit as st

from app.dashboard.api_client import DashboardApiError, get_json, post_json

st.title("Training Lab")
symbol = st.selectbox("Symbol", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h"])
source = st.text_input("Source", "demo-seed")
feature_set = st.selectbox("Feature set", ["v1", "v1.6_public"])

# ── Dataset Quality Report ───────────────────────────────────────────────────
st.subheader("📊 Dataset Quality")

try:
    quality = get_json(
        "/ui/dataset-quality",
        params={
            k: v
            for k, v in {
                "symbol": symbol,
                "timeframe": timeframe,
                "source": source or None,
                "feature_set_name": feature_set,
            }.items()
            if v is not None
        },
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", quality.get("total_rows", "—"))
    col2.metric("Features", quality.get("feature_columns", "—"))
    col3.metric(
        "Missing (avg)",
        f"{quality.get('overall_nan_ratio', 0.0) * 100:.1f}%",
    )
    col4.metric("Labels", quality.get("label_columns", "—"))

    with st.expander("NaN ratio by feature group"):
        nan_by_group = quality.get("nan_ratio_by_group", {})
        if nan_by_group:
            group_df = pd.DataFrame(
                [
                    {"Group": g, "NaN %": f"{v * 100:.1f}%"}
                    for g, v in sorted(nan_by_group.items())
                    if v > 0
                ]
            )
            if group_df.empty:
                st.success("All feature groups are complete (0 NaN).")
            else:
                st.dataframe(group_df, use_container_width=True, hide_index=True)

    with st.expander("Label distribution"):
        label_dist = quality.get("label_distribution", {})
        label_pct = quality.get("label_distribution_pct", {})
        missing_labels = quality.get("rows_missing_labels", 0)
        if label_dist:
            dist_df = pd.DataFrame(
                [
                    {"Label": k, "Count": v, "%": label_pct.get(k, 0)}
                    for k, v in sorted(label_dist.items())
                ]
            )
            st.dataframe(dist_df, use_container_width=True, hide_index=True)
        if missing_labels:
            st.warning(f"{missing_labels} rows will be dropped (missing labels).")

    with st.expander("Stale sparse rows"):
        stale = quality.get("stale_sparse_rows", {})
        if stale:
            stale_df = pd.DataFrame(
                [{"Group": g, "Stale rows": v} for g, v in sorted(stale.items()) if v > 0]
            )
            if stale_df.empty:
                st.success("No stale sparse rows detected.")
            else:
                st.dataframe(stale_df, use_container_width=True, hide_index=True)
        else:
            st.info("No sparse feature groups in this feature set.")

    with st.expander("Time range"):
        st.write(
            f"**First:** {quality.get('first_event_time', '—')} &nbsp;"
            f"**Last:** {quality.get('last_event_time', '—')}"
        )

except DashboardApiError as exc:
    st.warning(f"Dataset quality unavailable: {exc}")

st.divider()

# ── Train LightGBM ───────────────────────────────────────────────────────────
if st.button("Train LightGBM"):
    try:
        result = post_json(
            "/train/lightgbm",
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "source": source or None,
                "feature_set_version": feature_set,
            },
        )
        st.success(f"Model run {result['model_run_id']} saved.")
        st.json(result["metrics"])
    except DashboardApiError as exc:
        st.warning(str(exc))

try:
    runs = get_json("/ui/training-runs")["runs"]
except DashboardApiError as exc:
    st.warning(str(exc))
    st.stop()

st.subheader("Recent Training Runs")
if runs:
    rows = []
    for run in runs:
        rows.append(
            {
                "model_run_id": run["model_run_id"],
                "symbol": run["symbol"],
                "created_at": run["created_at"],
                "feature_set_version": run["feature_set_version"],
                "rows": run["metrics"].get("rows"),
                "features": run["metrics"].get("features"),
                "model": run["params"].get("model"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    latest_model_run_id = str(runs[0]["model_run_id"])
    try:
        evaluation = get_json(f"/models/{latest_model_run_id}/evaluation")
        baseline = evaluation.get("baseline_comparison", {})
        class_dist = evaluation.get("class_distribution", {})
        metrics = evaluation.get("metrics", {})

        def _num(value: object, default: float = 0.0) -> float:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    return default
            try:
                return float(str(value))
            except (TypeError, ValueError):
                return default

        st.subheader("Model Evaluation")
        c1, c2, c3 = st.columns(3)
        test_acc = _num(metrics.get("test_accuracy"))
        majority_acc = _num(baseline.get("baseline_majority_accuracy"))
        random_acc = _num(baseline.get("baseline_random_accuracy"))
        c1.metric("Test Acc", f"{test_acc:.4f}")
        c2.metric("Majority Baseline", f"{majority_acc:.4f}")
        c3.metric("Random Baseline", f"{random_acc:.4f}")

        c4, c5, c6 = st.columns(3)
        c4.metric(
            "Δ vs Majority",
            f"{_num(baseline.get('model_vs_majority_accuracy_delta')):+.4f}",
        )
        c5.metric(
            "Δ vs Random",
            f"{_num(baseline.get('model_vs_random_accuracy_delta')):+.4f}",
        )
        c6.metric("Target Classes", f"{int(_num(class_dist.get('target_class_count')))}")

        c7, c8 = st.columns(2)
        c7.metric("Max Class Ratio", f"{_num(class_dist.get('target_max_class_ratio')):.4f}")
        imbalance_flag = int(_num(class_dist.get("target_high_imbalance_flag")))
        c8.metric("High Imbalance", "yes" if imbalance_flag == 1 else "no")

        if test_acc <= majority_acc:
            st.warning("Model test accuracy does not beat majority baseline.")
    except DashboardApiError:
        st.info("Model evaluation report is unavailable for the latest run.")

    st.subheader("Feature Importance (Non-SHAP)")
    fi_top_n = st.slider("Top N features", min_value=5, max_value=50, value=20, step=5)
    fi_gain_col, fi_split_col = st.columns(2)
    try:
        fi_gain = get_json(
            f"/models/{latest_model_run_id}/feature-importance",
            params={"importance_type": "gain", "top_n": fi_top_n},
        )
        gain_rows = fi_gain.get("top_features", [])
        fi_gain_col.caption("Gain")
        if gain_rows:
            fi_gain_col.dataframe(
                pd.DataFrame(gain_rows)[["rank", "feature", "importance", "importance_share"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            fi_gain_col.info("No gain importance available.")
        if fi_gain.get("warning"):
            st.warning(str(fi_gain["warning"]))
    except DashboardApiError:
        fi_gain_col.info("Gain importance is unavailable.")

    try:
        fi_split = get_json(
            f"/models/{latest_model_run_id}/feature-importance",
            params={"importance_type": "split", "top_n": fi_top_n},
        )
        split_rows = fi_split.get("top_features", [])
        fi_split_col.caption("Split")
        if split_rows:
            fi_split_col.dataframe(
                pd.DataFrame(split_rows)[["rank", "feature", "importance", "importance_share"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            fi_split_col.info("No split importance available.")
    except DashboardApiError:
        fi_split_col.info("Split importance is unavailable.")

    st.subheader("Research Readiness")
    try:
        readiness = get_json(f"/models/{latest_model_run_id}/readiness-report")
        s1, s2 = st.columns(2)
        s1.metric("Status", str(readiness.get("readiness_status", "not_ready")))
        s2.metric("Score", int(readiness.get("readiness_score", 0)))

        blockers = readiness.get("blocking_reasons", [])
        warnings = readiness.get("warnings", [])
        if blockers:
            st.error("Blocking reasons:")
            for item in blockers:
                st.write(f"- {item}")
        if warnings:
            st.warning("Warnings:")
            for item in warnings:
                st.write(f"- {item}")

        checks = readiness.get("readiness_checks", {})
        if checks:
            check_rows = []
            for key, payload in checks.items():
                if isinstance(payload, dict):
                    check_rows.append(
                        {
                            "check": key,
                            "passed": bool(payload.get("passed", False)),
                            "points": int(payload.get("points", 0)),
                        }
                    )
            if check_rows:
                st.dataframe(pd.DataFrame(check_rows), use_container_width=True, hide_index=True)
    except DashboardApiError:
        st.info("Research readiness report is unavailable for the latest run.")
else:
    st.info("No model runs persisted yet.")
