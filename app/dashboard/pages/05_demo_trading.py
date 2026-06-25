from __future__ import annotations

import pandas as pd
import streamlit as st

from app.dashboard.api_client import (
    DashboardApiError,
    get_json,
    get_paper_lifecycle_events,
    get_paper_portfolio_audit,
    post_json,
)

st.title("Demo Trading")

try:
    demo_state = get_json("/ui/demo-state")
except DashboardApiError as exc:
    st.warning(str(exc))
    st.stop()

portfolio = demo_state["portfolio"]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Demo mode", str(demo_state["status"]))
col2.metric("Equity", f"{portfolio['equity']:.2f}")
col3.metric("Available margin", f"{portfolio['available_margin']:.2f}")
col4.metric("Open positions", str(len(portfolio["positions"])))

if st.button("Reset demo account"):
    try:
        result = post_json("/demo/reset")
        st.success(f"Demo account reset: {result['snapshot_id']}")
    except DashboardApiError as exc:
        st.warning(str(exc))

st.warning("Real order execution is disabled in v1.")

st.subheader("Paper/Demo Approval")
approval_note = "This does not execute orders."
try:
    latest = get_json("/models/latest?limit=1")
    model_runs = latest.get("model_runs", [])
    if model_runs:
        model_run_id = str(model_runs[0].get("model_run_id", ""))
        approval = get_json(f"/paper/approval/{model_run_id}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Status", str(approval.get("paper_demo_status", "blocked")))
        c2.metric("Approved", str(approval.get("approved_for_paper_demo", False)))
        c3.metric("Score", int(approval.get("approval_score", 0)))
        blockers = approval.get("blocking_reasons", [])
        warnings = approval.get("warnings", [])
        if blockers:
            st.error("Blockers:\n- " + "\n- ".join(str(item) for item in blockers))
        if warnings:
            st.warning("Warnings:\n- " + "\n- ".join(str(item) for item in warnings))
        st.caption(approval_note)
    else:
        st.info(f"No model runs available yet. {approval_note}")
except DashboardApiError:
    st.info(f"Paper/Demo approval is unavailable right now. {approval_note}")

st.subheader("Paper Simulation Audit")
st.caption(
    "Manual simulation only. Real execution disabled. Auto-loop not enabled. "
    "This panel is read-only diagnostics."
)
try:
    audit = get_json("/paper/simulation-audit", params={"limit": 50})
    summary = audit.get("summary", {})
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Total intents", int(summary.get("total_intents", 0)))
    a2.metric("Blocked intents", int(summary.get("blocked_intents", 0)))
    a3.metric("Candidate intents", int(summary.get("candidate_intents", 0)))
    a4.metric("Simulated intents", int(summary.get("simulated_intents", 0)))

    b1, b2, b3 = st.columns(3)
    b1.metric(
        "Candidate intents awaiting manual simulation",
        int(summary.get("unsimulated_candidate_intents", 0)),
    )
    b2.metric("Total paper orders", int(summary.get("total_orders", 0)))
    b3.metric("Total paper fills", int(summary.get("total_fills", 0)))

    blocking_reason_counts = audit.get("blocking_reason_counts", {})
    if isinstance(blocking_reason_counts, dict) and blocking_reason_counts:
        st.markdown("**Blocked intents by reason**")
        top_reasons = sorted(
            ((str(key), int(value)) for key, value in blocking_reason_counts.items()),
            key=lambda item: item[1],
            reverse=True,
        )[:5]
        st.table(pd.DataFrame(top_reasons, columns=["Reason", "Count"]))
    else:
        st.info("Blocked intents by reason: no data yet.")

    st.markdown("**Current paper portfolio summary**")
    portfolio_summary = audit.get("portfolio_summary", {})
    p1, p2, p3 = st.columns(3)
    p1.metric("Paper equity", f"{float(portfolio_summary.get('equity', 0.0)):.2f}")
    p2.metric(
        "Available margin",
        f"{float(portfolio_summary.get('available_margin', 0.0)):.2f}",
    )
    p3.metric("Open positions", int(portfolio_summary.get("positions_count", 0)))

    st.markdown("**Latest simulated orders**")
    latest_orders = audit.get("latest_orders", [])
    if latest_orders:
        st.dataframe(pd.DataFrame(latest_orders[:10]), use_container_width=True)
    else:
        st.info("No simulated orders yet.")

    st.markdown("**Latest simulated fills**")
    latest_fills = audit.get("latest_fills", [])
    if latest_fills:
        st.dataframe(pd.DataFrame(latest_fills[:10]), use_container_width=True)
    else:
        st.info("No simulated fills yet.")

    audit_warnings = audit.get("warnings", [])
    if isinstance(audit_warnings, list) and audit_warnings:
        st.warning("Warnings:\n- " + "\n- ".join(str(item) for item in audit_warnings))
except DashboardApiError:
    st.info("Paper simulation audit is unavailable right now.")

st.subheader("Paper Lifecycle Timeline / Audit Trail")
st.caption(
    "Manual simulation only. Real execution disabled. Auto-loop not enabled. "
    "Read-only lifecycle audit trail."
)
lifecycle_event_types = [
    "All",
    "intent_created",
    "intent_simulated",
    "intent_simulation_replayed",
    "paper_fill_closed",
    "paper_close_replayed",
]
lc1, lc2 = st.columns(2)
lifecycle_limit = int(
    lc1.number_input("Lifecycle events limit", min_value=1, max_value=500, value=100, step=10)
)
selected_event_type = lc2.selectbox("Lifecycle event type", options=lifecycle_event_types, index=0)
event_type_filter = None if selected_event_type == "All" else selected_event_type
try:
    lifecycle_payload = get_paper_lifecycle_events(
        limit=lifecycle_limit,
        event_type=event_type_filter,
    )
    lifecycle_events = lifecycle_payload.get("events", [])
    loaded_count = int(lifecycle_payload.get("count", len(lifecycle_events)))
    latest_event_time = (
        str(lifecycle_events[0].get("created_at", "n/a")) if lifecycle_events else "n/a"
    )

    l1, l2 = st.columns(2)
    l1.metric("Total events loaded", loaded_count)
    l2.metric("Latest event timestamp", latest_event_time)

    event_type_counts: dict[str, int] = {}
    for event in lifecycle_events:
        event_type = str(event.get("event_type", "unknown"))
        event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
    if event_type_counts:
        counts_table = sorted(event_type_counts.items(), key=lambda item: item[1], reverse=True)
        st.markdown("**Event type counts**")
        st.table(pd.DataFrame(counts_table, columns=["event_type", "count"]))
    else:
        st.info("No lifecycle event counts yet.")

    st.markdown("**Latest lifecycle events**")
    if lifecycle_events:
        latest_events_df = pd.DataFrame(lifecycle_events[:20])
        columns = [
            "created_at",
            "event_type",
            "entity_type",
            "entity_id",
            "related_entity_id",
        ]
        for column in columns:
            if column not in latest_events_df.columns:
                latest_events_df[column] = None
        st.dataframe(latest_events_df[columns], use_container_width=True)
    else:
        st.info("No lifecycle events yet.")
except DashboardApiError:
    st.info("Paper lifecycle events are unavailable right now.")

st.subheader("Paper Lifecycle Consistency / Portfolio Audit")
st.caption("Read-only audit panel. Manual simulation only. Real execution disabled.")
try:
    portfolio_audit = get_paper_portfolio_audit(limit=500)
    orphan_fills = portfolio_audit.get("orphan_fills", [])
    orphan_closes = portfolio_audit.get("orphan_closes", [])
    position_warnings = portfolio_audit.get("position_consistency_warnings", [])
    equity_consistency = portfolio_audit.get("equity_consistency", {})
    warnings = portfolio_audit.get("warnings", [])

    equity_checks = (
        equity_consistency.get("checks", []) if isinstance(equity_consistency, dict) else []
    )
    equity_mismatches = [
        check
        for check in equity_checks
        if isinstance(check, dict) and check.get("is_consistent") is False
    ]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Open fills", int(portfolio_audit.get("open_fill_count", 0)))
    c2.metric("Closed trades", int(portfolio_audit.get("closed_trade_count", 0)))
    c3.metric("Orphan fills", len(orphan_fills))
    c4.metric("Orphan closes", len(orphan_closes))

    d1, d2, d3 = st.columns(3)
    d1.metric("Position warnings", len(position_warnings))
    d2.metric("Equity mismatches", len(equity_mismatches))
    d3.metric("Warnings", len(warnings) if isinstance(warnings, list) else 0)

    is_healthy = (
        len(orphan_fills) == 0
        and len(orphan_closes) == 0
        and len(position_warnings) == 0
        and len(equity_mismatches) == 0
    )
    if is_healthy:
        st.success("Status: HEALTHY")
    else:
        st.warning("Status: NEEDS REVIEW")

    if orphan_fills:
        st.markdown("**Orphan fills**")
        st.dataframe(pd.DataFrame(orphan_fills), use_container_width=True)
    if orphan_closes:
        st.markdown("**Orphan closes**")
        st.dataframe(pd.DataFrame(orphan_closes), use_container_width=True)
    if position_warnings:
        st.markdown("**Position consistency warnings**")
        warnings_df = pd.DataFrame([{"warning": str(item)} for item in position_warnings])
        st.dataframe(warnings_df, use_container_width=True)
    if equity_mismatches:
        st.markdown("**Equity consistency mismatches**")
        st.dataframe(pd.DataFrame(equity_mismatches), use_container_width=True)
except DashboardApiError:
    st.info("Paper portfolio audit is unavailable right now.")

st.subheader("Paper Track Record / Investor Metrics")
st.caption(
    "Read-only investor metrics for paper trading history. "
    "Manual simulation only. Real execution disabled."
)
st.caption("Manual close simulation API: POST /paper/fills/{fill_id}/close (no auto-loop).")
try:
    track = get_json("/paper/track-record", params={"limit": 500})
    summary = track.get("summary", {})
    performance = track.get("performance_metrics", {})
    risk = track.get("risk_metrics", {})

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Paper equity", f"{float(summary.get('paper_equity') or 0.0):.2f}")
    t2.metric("Total PnL", str(performance.get("total_pnl")))
    t3.metric("Total return %", str(performance.get("total_return_pct")))
    t4.metric("Win rate", str(performance.get("win_rate")))

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Profit factor", str(performance.get("profit_factor")))
    r2.metric("Max drawdown", str(risk.get("max_drawdown")))
    r3.metric("Best day PnL", str(risk.get("best_day_pnl")))
    r4.metric("Worst day PnL", str(risk.get("worst_day_pnl")))

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Open trades", int(summary.get("open_trades", 0)))
    s2.metric("Closed trades", int(summary.get("closed_trades", 0)))
    s3.metric("Total notional", str(summary.get("total_notional")))
    s4.metric("Fees paid", str(summary.get("total_fees_paid")))

    st.markdown("**Latest paper trades**")
    latest_trades = track.get("latest_trades", [])
    if latest_trades:
        st.dataframe(pd.DataFrame(latest_trades[:10]), use_container_width=True)
    else:
        st.info("No paper trades yet.")

    st.markdown("**Daily summary**")
    daily_summary = track.get("daily_summary", [])
    if daily_summary:
        st.dataframe(pd.DataFrame(daily_summary[-10:]), use_container_width=True)
    else:
        st.info("No daily summary yet.")

    st.markdown("**Monthly summary**")
    monthly_summary = track.get("monthly_summary", [])
    if monthly_summary:
        st.dataframe(pd.DataFrame(monthly_summary), use_container_width=True)
    else:
        st.info("No monthly summary yet.")

    track_warnings = track.get("warnings", [])
    if isinstance(track_warnings, list) and track_warnings:
        st.warning("Warnings:\n- " + "\n- ".join(str(item) for item in track_warnings))
except DashboardApiError:
    st.info("Paper track record is unavailable right now.")

st.subheader("Paper Orders")
orders = demo_state["orders"]
if orders:
    st.dataframe(pd.DataFrame(orders), use_container_width=True)
else:
    st.info("No paper orders yet.")

st.subheader("Paper Fills")
fills = demo_state["fills"]
if fills:
    st.dataframe(pd.DataFrame(fills), use_container_width=True)
else:
    st.info("No paper fills yet.")
