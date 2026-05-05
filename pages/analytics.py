import streamlit as st
from auth import check_auth

check_auth()

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from modules.analytics_redesigned import *
from modules.smote_extension import (
    smote_cv_comparison,
    smote_full_model,
    prepare_tree_data,   # re-export from analytics_redesigned if needed
)


# ──────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ──────────────────────────────────────────────────────────────────────────

st.title("Prospecting Analytics Dashboard")
st.markdown(
    "Data-driven insights into prospecting performance: funnel analysis, "
    "segment breakdowns, decision-tree modeling, and SMOTE imbalance comparison."
)

# Session state initialisation
for key, val in [
    ("model_trained",    False),
    ("model_data",       None),
    ("smote_run",        False),
    ("smote_cv_data",    None),
]:
    if key not in st.session_state:
        st.session_state[key] = val


# ──────────────────────────────────────────────────────────────────────────
# SIDEBAR — DATA UPLOAD
# ──────────────────────────────────────────────────────────────────────────

st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader(
    "Upload prospects CSV",
    type=["csv"],
    help=(
        "Required columns: Lead Type, Company, Industry, Size, "
        "Decision Level, Contact Channel, Replied, Meeting, Converted"
    ),
)

if uploaded_file is None:
    st.info("👈 Please upload a CSV file to get started")
    st.stop()

df = pd.read_csv(uploaded_file)
df, warnings_list = load_and_validate(df)

if warnings_list:
    st.sidebar.warning("⚠️ Data Warnings:\n" + "\n".join(warnings_list))
st.sidebar.success(f"✓ Loaded {len(df)} prospects")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1  CONVERSION FUNNEL
# ══════════════════════════════════════════════════════════════════════════════

st.header("Conversion Funnel")
funnel = funnel_stats(df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Prospects Contacted", funnel["total"])
c2.metric("Replied",          funnel["replied"],   f"{funnel['reply_rate']}%")
c3.metric("Meetings Scheduled", funnel["meeting"], f"{funnel['meeting_rate']}%")
c4.metric("Converted",        funnel["converted"], f"{funnel['conversion_rate']}%")

st.subheader("Conversion Funnel Waterfall")
fig_funnel = px.funnel(
    pd.DataFrame({
        "Stage": ["Contacted", "Replied", "Meeting", "Converted"],
        "Count": [funnel["total"], funnel["replied"],
                  funnel["meeting"], funnel["converted"]],
    }),
    x="Count", y="Stage",
    color="Stage",
    title="Prospect Conversion Funnel",
    color_discrete_sequence=px.colors.sequential.Blues_r,
)
fig_funnel.update_layout(height=400)
st.plotly_chart(fig_funnel, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2  SEGMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

st.header("Segment Analysis (Client vs Partner)")
seg_comp = segment_comparison(df)

if seg_comp:
    cols = st.columns(len(seg_comp))
    for idx, (segment, data) in enumerate(seg_comp.items()):
        with cols[idx]:
            st.subheader(segment)
            st.metric("Count",        data["count"])
            st.metric("Reply Rate",   f"{data['funnel']['reply_rate']}%")
            st.metric("Meeting Rate", f"{data['funnel']['meeting_rate']}%")
            if data["top_channel"] is not None:
                tc = data["top_channel"]
                st.caption(
                    f"**Top Channel:** {tc['Contact Channel']} "
                    f"({tc['Reply rate %']}%)"
                )
            if data["top_decision_level"] is not None:
                td = data["top_decision_level"]
                st.caption(
                    f"**Top Decision Level:** {td['Decision Level']} "
                    f"({td['Reply rate %']}%)"
                )
else:
    st.info("Add 'Lead Type' column (Client / Partner) to see segment analysis")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3  PERFORMANCE BREAKDOWNS
# ══════════════════════════════════════════════════════════════════════════════

st.header("Performance Breakdowns")
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["By Channel", "By Industry", "By Decision Level", "By Size", "By Segment+Channel"]
)

with tab1:
    st.subheader("Reply Rates by Contact Channel")
    ch_bd = channel_breakdown(df)
    if not ch_bd.empty:
        top = ch_bd.iloc[0]
        st.success(
            f"**{top['Contact Channel']}** is most effective: "
            f"{top['Reply rate %']}% reply rate "
            f"({int(top['Replied'])}/{int(top['Total'])} prospects)"
        )
        st.dataframe(
            ch_bd[["Contact Channel", "Total", "Replied", "Reply rate %",
                   "Meeting", "Meeting rate %", "Converted"]].round(1),
            use_container_width=True, hide_index=True,
        )

with tab2:
    st.subheader("Reply Rates by Industry")
    ind_bd = industry_breakdown(df)
    if not ind_bd.empty:
        top = ind_bd.iloc[0]
        st.success(
            f"**{top['Industry']}** most responsive: {top['Reply rate %']}% reply rate"
        )
        st.dataframe(
            ind_bd[["Industry", "Total", "Replied", "Reply rate %",
                    "Meeting", "Meeting rate %"]].round(1),
            use_container_width=True, hide_index=True,
        )
        st.plotly_chart(
            px.bar(ind_bd.head(10), x="Industry", y="Reply rate %",
                   title="Top Industries by Reply Rate",
                   color="Reply rate %", color_continuous_scale="Blues"),
            use_container_width=True,
        )

with tab3:
    st.subheader("Reply Rates by Decision Level")
    dl_bd = decision_level_breakdown(df)
    if not dl_bd.empty:
        top, worst = dl_bd.iloc[0], dl_bd.iloc[-1]
        st.success(
            f"**{top['Decision Level']}** level most responsive: "
            f"{top['Reply rate %']}% vs {worst['Decision Level']} at "
            f"{worst['Reply rate %']}%"
        )
        st.dataframe(
            dl_bd[["Decision Level", "Total", "Replied", "Reply rate %",
                   "Meeting", "Meeting rate %"]].round(1),
            use_container_width=True, hide_index=True,
        )
        st.plotly_chart(
            px.bar(dl_bd, x="Decision Level", y="Reply rate %",
                   title="Decision Level Impact on Reply Rate",
                   color="Reply rate %", color_continuous_scale="Blues"),
            use_container_width=True,
        )

with tab4:
    st.subheader("Reply Rates by Company Size")
    sz_bd = size_breakdown(df)
    if not sz_bd.empty:
        st.dataframe(
            sz_bd[["Size bucket", "Total", "Replied", "Reply rate %",
                   "Meeting", "Meeting rate %"]].round(1),
            use_container_width=True, hide_index=True,
        )

with tab5:
    st.subheader("Reply Rates by Segment + Channel")
    seg_bd = segment_breakdown(df)
    if not seg_bd.empty:
        st.dataframe(
            seg_bd[["Lead Type", "Contact Channel", "Total", "Replied",
                    "Reply rate %", "Meeting rate %"]].round(1),
            use_container_width=True, hide_index=True,
        )
        pivot = seg_bd.pivot_table(
            values="Reply rate %", index="Lead Type",
            columns="Contact Channel", aggfunc="first",
        )
        st.plotly_chart(
            px.imshow(pivot, text_auto=True,
                      title="Reply Rate Heatmap: Segment × Channel",
                      color_continuous_scale="Blues"),
            use_container_width=True,
        )
    else:
        st.info("Add 'Lead Type' column for segment+channel analysis")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4  DECISION TREE MODELING
# ══════════════════════════════════════════════════════════════════════════════

st.header("Decision Tree Modeling")
st.markdown(
    "Train segment-specific decision trees to predict prospect reply likelihood. "
    "The model identifies which features (Decision Level, Industry, Channel, Size, Month) "
    "matter most for each segment."
)

col1, col2, col3 = st.columns(3)
with col1:
    segment_select = st.selectbox(
        "Select segment to model:",
        ["All Data", "Client", "Partner"],
        help="All Data: Use all prospects. Client: Direct buyers. Partner: Channel partners",
    )
with col2:
    target_select = st.selectbox(
        "Predict which outcome:",
        ["Replied", "Meeting", "Converted"],
        help="Replied: Any response. Meeting: Agreed to call. Converted: Sale or LOI",
    )
with col3:
    if st.button("Train Model", use_container_width=True):
        st.session_state.model_trained = True
        seg_param = None if segment_select == "All Data" else segment_select
        st.session_state.model_data = model_summary(
            df, segment=seg_param, target=target_select
        )

if st.session_state.model_trained and st.session_state.model_data:
    summary = st.session_state.model_data

    if "error" in summary:
        st.error(f"Model Error: {summary['error']}")
    else:
        st.subheader(f"Results: {summary['segment']} → {summary['target']}")

        cv = summary["cv_results"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("F1 Score",    f"{cv['f1_mean']:.3f}",        f"±{cv['f1_std']:.3f}")
        c2.metric("Precision",   f"{cv['precision_mean']:.3f}")
        c3.metric("Recall",      f"{cv['recall_mean']:.3f}")
        c4.metric("Sample Size", f"{cv['n_samples']}")
        st.caption(
            f"{cv['n_samples']} prospects | "
            f"{cv['positive_rate']}% positive outcomes | "
            f"{cv['n_folds']}-fold CV"
        )

        cm = summary["confusion_matrix"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("True Positives",  cm["true_positives"],  "Correct replies")
        c2.metric("False Positives", cm["false_positives"],  "Predicted but didn't reply")
        c3.metric("False Negatives", cm["false_negatives"],  "Missed replies")
        c4.metric("True Negatives",  cm["true_negatives"],   "Correct non-replies")

        st.info(
            f"**Precision:** {cm['precision']:.1%} of flagged prospects actually replied  \n"
            f"**Recall:** {cm['recall']:.1%} of all replies were identified by the model"
        )

        fi = summary["feature_importance"]
        col_left, col_right = st.columns([2, 1])
        with col_left:
            fig_fi = px.bar(
                fi.head(10), x="Importance %", y="Feature", orientation="h",
                title="Top 10 Features Driving Predictions",
                color="Importance %", color_continuous_scale="Blues",
            )
            fig_fi.update_layout(yaxis={"categoryorder": "total ascending"}, height=400)
            st.plotly_chart(fig_fi, use_container_width=True)
        with col_right:
            st.write("**Top 3 Features:**")
            for i, row in enumerate(summary["top_3_features"], 1):
                st.write(f"{i}. {row['Feature']}: {row['Importance %']:.1f}%")

        with st.expander("View Full Decision Rules (If-Then Logic)"):
            st.code(summary["decision_rules"], language="text")

        top_feat = fi.iloc[0]
        if cm["precision"] > 0.5:
            st.write(
                f"**{top_feat['Feature']}** is the strongest predictor "
                f"({top_feat['Importance %']:.1f}%). "
                f"High precision ({cm['precision']:.1%}) — your top-scored prospects "
                "are likely to reply. Prioritise them."
            )
        else:
            st.write(
                f"**{top_feat['Feature']}** is the strongest predictor "
                f"({top_feat['Importance %']:.1f}%). "
                f"Low precision ({cm['precision']:.1%}) — the model flags many "
                "non-replies. Collect more outcome data or enrich features."
            )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5  SMOTE — CLASS IMBALANCE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

st.header("Class Imbalance: SMOTE Comparison")

st.markdown(
    """
    Prospecting datasets are **heavily imbalanced** — far more non-replies (0) than
    replies (1). This can bias a Decision Tree toward always predicting "no reply."

    **SMOTE** (Synthetic Minority Over-sampling Technique) generates synthetic positive
    examples by interpolating between existing minority samples, giving the model a
    more balanced view of both classes.

    > **Important:** SMOTE is applied **only on the training fold** inside each
    > cross-validation split. The test set is never touched — this prevents data
    > leakage and ensures honest evaluation.
    """
)

# ── controls ──────────────────────────────────────────────────────────────────
smote_col1, smote_col2, smote_col3 = st.columns(3)

with smote_col1:
    smote_segment = st.selectbox(
        "Segment:",
        ["All Data", "Client", "Partner"],
        key="smote_segment",
    )
with smote_col2:
    smote_target = st.selectbox(
        "Predict:",
        ["Replied", "Meeting", "Converted"],
        key="smote_target",
    )
with smote_col3:
    smote_folds = st.selectbox(
        "CV folds:",
        [5, 10],
        index=0,
        key="smote_folds",
        help="5-fold is recommended for small datasets (< 100 records)",
    )

if st.button("Run SMOTE Comparison", use_container_width=True, type="primary"):
    seg_param = None if smote_segment == "All Data" else smote_segment
    X, y = prepare_tree_data(df, segment=seg_param, target=smote_target)

    if X is None:
        st.error("Not enough data for the selected segment/target.")
    else:
        with st.spinner("Training models with and without SMOTE across folds…"):
            result = smote_cv_comparison(X, y, n_splits=int(smote_folds))

        if "error" in result:
            st.error(result["error"])
        else:
            st.session_state.smote_cv_data = result
            st.session_state.smote_run = True

# ── results display ───────────────────────────────────────────────────────────
if st.session_state.smote_run and st.session_state.smote_cv_data:
    res = st.session_state.smote_cv_data

    if "error" in res:
        st.error(res["error"])
    else:
        wo = res["without_smote"]
        wi = res["with_smote"]
        cmp = res["comparison"]

        # context banner
        st.info(
            f"**Dataset:** {res['n_samples']} prospects | "
            f"**Positive rate:** {res['positive_rate']}% | "
            f"**CV folds used:** {wo.get('n_folds_used', res['n_splits'])}"
        )

        # ── side-by-side metric panels ────────────────────────────────────
        st.subheader("Side-by-Side Comparison")

        left, right = st.columns(2)

        def _render_panel(col, label: str, data: dict, highlight_recall: bool = True):
            col.markdown(f"### {label}")
            metrics_map = [
                ("Accuracy",  "accuracy_mean",  "accuracy_std"),
                ("Precision", "precision_mean", "precision_std"),
                ("Recall ⭐", "recall_mean",    "recall_std"),
                ("F1 Score",  "f1_mean",        "f1_std"),
            ]
            for display, mean_key, std_key in metrics_map:
                mean_val = data.get(mean_key, 0)
                std_val  = data.get(std_key, 0)
                col.metric(
                    label=display,
                    value=f"{mean_val:.1f}%",
                    delta=f"±{std_val:.1f}% std",
                    delta_color="off",
                )
            col.markdown("**Confusion matrix (pooled folds)**")
            col.markdown(
                f"TP `{data.get('tp', 0)}` · FP `{data.get('fp', 0)}` · "
                f"FN `{data.get('fn', 0)}` · TN `{data.get('tn', 0)}`"
            )

        with left:
            _render_panel(left, "Model WITHOUT SMOTE", wo)
        with right:
            _render_panel(right, "Model WITH SMOTE", wi)

        # ── delta summary table ───────────────────────────────────────────
        st.subheader("Impact of SMOTE (Δ vs baseline)")

        rows = []
        for metric, vals in cmp.items():
            delta = vals["delta"]
            rows.append({
                "Metric":           f"{metric.capitalize()}",
                "Without SMOTE":    f"{vals['without']:.1f}%",
                "With SMOTE":       f"{vals['with']:.1f}%",
                "Δ Change":         f"{'▲' if delta > 0 else '▼' if delta < 0 else '='} {abs(delta):.1f}%",
                "Interpretation":   "Improved ✅" if delta > 0 else
                                    ("Declined ⚠️" if delta < 0 else "No change —"),
            })

        delta_df = pd.DataFrame(rows)
        st.dataframe(delta_df, use_container_width=True, hide_index=True)

        # ── visual: grouped bar chart ─────────────────────────────────────
        chart_df = pd.DataFrame({
            "Metric":  ["Accuracy", "Precision", "Recall", "F1"],
            "Without SMOTE": [
                wo.get("accuracy_mean",  0), wo.get("precision_mean", 0),
                wo.get("recall_mean",    0), wo.get("f1_mean",        0),
            ],
            "With SMOTE": [
                wi.get("accuracy_mean",  0), wi.get("precision_mean", 0),
                wi.get("recall_mean",    0), wi.get("f1_mean",        0),
            ],
        }).melt(id_vars="Metric", var_name="Model", value_name="Score (%)")

        fig_cmp = px.bar(
            chart_df,
            x="Metric", y="Score (%)", color="Model",
            barmode="group",
            title="Decision Tree Performance: With vs Without SMOTE",
            color_discrete_sequence=["#5B8DB8", "#F4A261"],
            text_auto=".1f",
        )
        fig_cmp.update_traces(textposition="outside")
        fig_cmp.update_layout(height=420, legend_title_text="")
        st.plotly_chart(fig_cmp, use_container_width=True)

        # ── recall callout (most important for prospecting) ───────────────
        recall_cmp = cmp.get("recall", {})
        recall_delta = recall_cmp.get("delta", 0)

        st.subheader("Recall: Why It Matters Most for Prospecting")
        if recall_delta > 0:
            st.success(
                f"**SMOTE improved Recall by {recall_delta:.1f} percentage points** "
                f"({recall_cmp['without']:.1f}% → {recall_cmp['with']:.1f}%). "
                "This means the model identifies more true replies — fewer high-intent "
                "prospects slip through undetected."
            )
        elif recall_delta < 0:
            st.warning(
                f"**SMOTE reduced Recall by {abs(recall_delta):.1f} percentage points** "
                f"({recall_cmp['without']:.1f}% → {recall_cmp['with']:.1f}%). "
                "With very few positive examples, synthetic samples may not capture "
                "real behavioral patterns — see limitations below."
            )
        else:
            st.info(
                "SMOTE did not meaningfully change Recall. "
                "This is common with very small minority classes "
                f"(positive rate: {res['positive_rate']}%)."
            )

        # ── honest limitations box ────────────────────────────────────────
        with st.expander("⚠️ Limitations of SMOTE on this dataset"):
            st.markdown(
                f"""
                **Sample size:** {res['n_samples']} total observations with
                {res['positive_rate']}% positive rate
                (~{round(res['n_samples'] * res['positive_rate'] / 100)} replies).

                **Synthetic ≠ real:** SMOTE interpolates between existing minority
                samples. With very few real replies, synthetic samples are close
                neighbours of the same few points — they introduce little new
                information about true reply behaviour.

                **Per-fold minority count:** With {res['n_splits']}-fold CV,
                each training fold contains roughly
                {round(res['n_samples'] * res['positive_rate'] / 100 * (1 - 1/res['n_splits']))}
                positive examples. Below ~5, SMOTE is limited to
                `k_neighbors = minority_count − 1`, further constraining diversity.

                **Interpretation:** These results are **exploratory** and should
                not be used as definitive performance estimates. Collect more
                labelled data (target: ≥ 30 positive outcomes) before deploying
                a scored prospecting model in production.
                """
            )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6  TEMPORAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

st.header("Temporal Trends")
temporal = temporal_analysis(df)
col1, col2 = st.columns(2)

with col1:
    st.subheader("Weekly Reply Rate Trend")
    weekly = temporal["weekly_breakdown"]
    if not weekly.empty:
        st.plotly_chart(
            px.line(
                weekly, x="Week Label", y="Reply rate %", markers=True,
                title="Reply Rate by Week",
                color_discrete_sequence=["#1f77b4"],
            ),
            use_container_width=True,
        )
        st.dataframe(
            weekly[["Week Label", "Total", "Replied", "Reply rate %", "Meeting rate %"]],
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("Add 'Contact Date' column for weekly analysis")

with col2:
    st.subheader("Monthly Reply Rate Trend")
    monthly = temporal["monthly_breakdown"]
    if not monthly.empty:
        st.plotly_chart(
            px.bar(
                monthly, x="Contact Month", y="Reply rate %",
                title="Reply Rate by Month",
                color="Reply rate %", color_continuous_scale="Blues",
            ),
            use_container_width=True,
        )
        st.dataframe(
            monthly[["Contact Month", "Total", "Replied", "Reply rate %"]],
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("Add 'Contact Date' column for monthly analysis")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7  KEY INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════

st.header("Key Insights")
for i, insight in enumerate(generate_insights(df), 1):
    st.write(f"{i}. {insight}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8  STRATEGIC RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════

st.header("Strategic Recommendations")
dl_bd = decision_level_breakdown(df)
ch_bd = channel_breakdown(df)
ind_bd = industry_breakdown(df)

if not dl_bd.empty and not ch_bd.empty:
    best_dl  = dl_bd.iloc[0]
    worst_dl = dl_bd.iloc[-1]
    best_ch  = ch_bd.iloc[0]

    rec = (
        f"### For Maximum Reply Rate\n\n"
        f"**Target Decision Level:** {best_dl['Decision Level']}  \n"
        f"- Reply rate: {best_dl['Reply rate %']}% "
        f"({int(best_dl['Replied'])}/{int(best_dl['Total'])} prospects)  \n"
        f"- vs {worst_dl['Decision Level']}: {worst_dl['Reply rate %']}% "
        f"(gap: {best_dl['Reply rate %'] - worst_dl['Reply rate %']:.1f}%)  \n"
        f"- **Action:** Allocate 70% of prospecting efforts to "
        f"{best_dl['Decision Level']} level\n\n"
        f"**Preferred Channel:** {best_ch['Contact Channel']}  \n"
        f"- Reply rate: {best_ch['Reply rate %']}% "
        f"({int(best_ch['Replied'])}/{int(best_ch['Total'])} prospects)  \n"
        f"- **Action:** Use {best_ch['Contact Channel']} as primary channel\n\n"
        f"**Industries to Prioritise:**"
    )
    if not ind_bd.empty:
        for idx, (_, row) in enumerate(ind_bd.head(3).iterrows(), 1):
            rec += f"\n{idx}. {row['Industry']}: {row['Reply rate %']}% reply rate"

    rec += (
        f"\n\n### Expected Performance\n"
        f"- Baseline reply rate: {funnel['reply_rate']}% (all prospects)  \n"
        f"- Optimised reply rate: 40–50% (top 20% by score)  \n"
        f"- Improvement factor: 1.5–1.7×"
    )
    st.markdown(rec)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9  EXPORT
# ══════════════════════════════════════════════════════════════════════════════

st.header("Export Results")
c1, c2, c3 = st.columns(3)
with c1:
    st.download_button(
        "Channel Breakdown (CSV)",
        channel_breakdown(df).to_csv(index=False),
        "channel_breakdown.csv", "text/csv",
    )
with c2:
    st.download_button(
        "Decision Level Breakdown (CSV)",
        decision_level_breakdown(df).to_csv(index=False),
        "decision_level_breakdown.csv", "text/csv",
    )
with c3:
    wk = weekly_analysis(df)
    if not wk.empty:
        st.download_button(
            "Weekly Analysis (CSV)",
            wk.to_csv(index=False),
            "weekly_analysis.csv", "text/csv",
        )

st.divider()

try:
    from io import BytesIO
    import openpyxl  # noqa: F401 — just checking it's installed

    export_sheets = {
        "Funnel":         pd.DataFrame([funnel]),
        "Channel":        channel_breakdown(df),
        "Industry":       industry_breakdown(df),
        "Decision Level": decision_level_breakdown(df),
        "Size":           size_breakdown(df),
        "Weekly":         weekly_analysis(df),
    }

    seg_rows = [
        {"Segment": seg, "Count": d["count"],
         "Reply %": d["funnel"]["reply_rate"],
         "Meeting %": d["funnel"]["meeting_rate"]}
        for seg, d in segment_comparison(df).items()
    ]
    if seg_rows:
        export_sheets["Segments"] = pd.DataFrame(seg_rows)

    # add SMOTE results if available
    if st.session_state.smote_run and st.session_state.smote_cv_data:
        r = st.session_state.smote_cv_data
        if "comparison" in r:
            smote_rows = [
                {"Metric": m.capitalize(),
                 "Without SMOTE %": v["without"],
                 "With SMOTE %": v["with"],
                 "Delta %": v["delta"]}
                for m, v in r["comparison"].items()
            ]
            export_sheets["SMOTE Comparison"] = pd.DataFrame(smote_rows)

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, sdf in export_sheets.items():
            if not sdf.empty:
                sdf.to_excel(writer, sheet_name=name, index=False)
    buf.seek(0)

    st.download_button(
        "Download All Analytics (Excel)",
        buf.getvalue(),
        "VEEP_Analytics_Complete.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
except ImportError:
    st.warning("openpyxl not installed — CSV exports above are available.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "**VEEP Prospecting Framework — Analytics Module** | "
    "Decision-tree-based prospecting with SMOTE imbalance correction | "
    "Segment-specific modeling (Clients vs Partners)"
)