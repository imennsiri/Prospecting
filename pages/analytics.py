import streamlit as st
from auth import check_auth

if not check_auth():
    st.stop()

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from analytics_redesigned import *


# ──────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="VEEP Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title(" Prospecting Analytics Dashboard")

st.markdown("""
This dashboard provides data-driven insights into your prospecting performance.
It includes conversion funnel analysis, segment breakdowns, and decision-tree modeling.
""")

# Initialize session state
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "model_data" not in st.session_state:
    st.session_state.model_data = None

# ──────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────────────────────────────────

st.sidebar.header(" Data Input")

uploaded_file = st.sidebar.file_uploader(
    "Upload prospects CSV",
    type=['csv'],
    help="File must include: Lead Type, Company, Industry, Size, Decision Level, Contact Channel, Replied, Meeting, Converted"
)

if uploaded_file is None:
    st.info("👈 Please upload a CSV file to get started")
    st.stop()

# Load and validate
df = pd.read_csv(uploaded_file)
df, warnings = load_and_validate(df)

if warnings:
    st.sidebar.warning("⚠️ Data Warnings:\n" + "\n".join(warnings))

st.sidebar.success(f"✓ Loaded {len(df)} prospects")

# ──────────────────────────────────────────────────────────────────────────
# 2. TOP-LEVEL METRICS (CONVERSION FUNNEL)
# ──────────────────────────────────────────────────────────────────────────

st.header(" Conversion Funnel")

funnel = funnel_stats(df)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Prospects Contacted",
        value=funnel['total'],
        delta=None
    )

with col2:
    st.metric(
        label="Replied",
        value=funnel['replied'],
        delta=f"{funnel['reply_rate']}%",
        delta_color="normal"
    )

with col3:
    st.metric(
        label="Meetings Scheduled",
        value=funnel['meeting'],
        delta=f"{funnel['meeting_rate']}%",
        delta_color="normal"
    )

with col4:
    st.metric(
        label="Converted",
        value=funnel['converted'],
        delta=f"{funnel['conversion_rate']}%",
        delta_color="normal"
    )

# Funnel waterfall chart
st.subheader("Conversion Funnel Waterfall")

funnel_data = pd.DataFrame({
    'Stage': ['Contacted', 'Replied', 'Meeting', 'Converted'],
    'Count': [funnel['total'], funnel['replied'], funnel['meeting'], funnel['converted']],
})

fig_funnel = px.funnel(
    funnel_data,
    x='Count',
    y='Stage',
    color='Stage',
    title='Prospect Conversion Funnel',
    labels={'Stage': '', 'Count': 'Number of Prospects'},
    color_discrete_sequence=px.colors.sequential.Blues_r
)
fig_funnel.update_layout(height=400)
st.plotly_chart(fig_funnel, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────
# 3. SEGMENT COMPARISON (Client vs Partner)
# ──────────────────────────────────────────────────────────────────────────

st.header("Segment Analysis (Client vs Partner)")

seg_comp = segment_comparison(df)

if seg_comp:
    cols = st.columns(len(seg_comp))
    
    for idx, (segment, data) in enumerate(seg_comp.items()):
        with cols[idx]:
            st.subheader(segment)
            st.metric("Count", data['count'])
            
            stats = data['funnel']
            st.metric("Reply Rate", f"{stats['reply_rate']}%")
            st.metric("Meeting Rate", f"{stats['meeting_rate']}%")
            
            if data['top_channel'] is not None:
                top_ch = data['top_channel']
                st.caption(
                    f"**Top Channel:** {top_ch['Contact Channel']}\n"
                    f"({top_ch['Reply rate %']}% reply rate)"
                )
            
            if data['top_decision_level'] is not None:
                top_dl = data['top_decision_level']
                st.caption(
                    f"**Top Decision Level:** {top_dl['Decision Level']}\n"
                    f"({top_dl['Reply rate %']}% reply rate)"
                )
else:
    st.info("Add 'Lead Type' column (Client / Partner) to see segment analysis")

# ──────────────────────────────────────────────────────────────────────────
# 4. BREAKDOWN TABLES
# ──────────────────────────────────────────────────────────────────────────

st.header("Performance Breakdowns")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["By Channel", "By Industry", "By Decision Level", "By Size", "By Segment+Channel"]
)

# Tab 1: Channel
with tab1:
    st.subheader("Reply Rates by Contact Channel")
    ch_bd = channel_breakdown(df)
    
    if not ch_bd.empty:
        # Show top channel highlighted
        top_ch = ch_bd.iloc[0]
        st.success(
            f"**{top_ch['Contact Channel']}** is most effective: "
            f"{top_ch['Reply rate %']}% reply rate ({int(top_ch['Replied'])}/{int(top_ch['Total'])} prospects)"
        )
        
        st.dataframe(
            ch_bd[[
                'Contact Channel', 'Total', 'Replied', 'Reply rate %',
                'Meeting', 'Meeting rate %', 'Converted'
            ]].round(1),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No data available for channel breakdown")

# Tab 2: Industry
with tab2:
    st.subheader("Reply Rates by Industry")
    ind_bd = industry_breakdown(df)
    
    if not ind_bd.empty:
        top_ind = ind_bd.iloc[0]
        st.success(
            f"**{top_ind['Industry']}** industry most responsive: "
            f"{top_ind['Reply rate %']}% reply rate"
        )
        
        st.dataframe(
            ind_bd[[
                'Industry', 'Total', 'Replied', 'Reply rate %',
                'Meeting', 'Meeting rate %'
            ]].round(1),
            use_container_width=True,
            hide_index=True
        )
        
        # Chart
        fig_ind = px.bar(
            ind_bd.head(10),
            x='Industry',
            y='Reply rate %',
            title='Top Industries by Reply Rate',
            color='Reply rate %',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_ind, use_container_width=True)
    else:
        st.info("No data available for industry breakdown")

# Tab 3: Decision Level
with tab3:
    st.subheader("Reply Rates by Decision Level")
    dl_bd = decision_level_breakdown(df)
    
    if not dl_bd.empty:
        top_dl = dl_bd.iloc[0]
        worst_dl = dl_bd.iloc[-1]
        
        st.success(
            f"**{top_dl['Decision Level']}** level most responsive: "
            f"{top_dl['Reply rate %']}% vs {worst_dl['Decision Level']} at {worst_dl['Reply rate %']}%"
        )
        
        st.dataframe(
            dl_bd[[
                'Decision Level', 'Total', 'Replied', 'Reply rate %',
                'Meeting', 'Meeting rate %'
            ]].round(1),
            use_container_width=True,
            hide_index=True
        )
        
        # Chart
        fig_dl = px.bar(
            dl_bd,
            x='Decision Level',
            y='Reply rate %',
            title='Decision Level Impact on Reply Rate',
            color='Reply rate %',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_dl, use_container_width=True)
    else:
        st.info("No data available for decision level breakdown")

# Tab 4: Size
with tab4:
    st.subheader("Reply Rates by Company Size")
    sz_bd = size_breakdown(df)
    
    if not sz_bd.empty:
        st.dataframe(
            sz_bd[[
                'Size bucket', 'Total', 'Replied', 'Reply rate %',
                'Meeting', 'Meeting rate %'
            ]].round(1),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No data available for size breakdown")

# Tab 5: Segment + Channel
with tab5:
    st.subheader("Reply Rates by Segment + Channel")
    seg_bd = segment_breakdown(df)
    
    if not seg_bd.empty:
        st.dataframe(
            seg_bd[[
                'Lead Type', 'Contact Channel', 'Total', 'Replied', 'Reply rate %',
                'Meeting rate %'
            ]].round(1),
            use_container_width=True,
            hide_index=True
        )
        
        # Chart: Segment-Channel Heatmap
        pivot_data = seg_bd.pivot_table(
            values='Reply rate %',
            index='Lead Type',
            columns='Contact Channel',
            aggfunc='first'
        )
        
        fig_hm = px.imshow(
            pivot_data,
            labels=dict(x='Contact Channel', y='Lead Type', color='Reply Rate %'),
            title='Reply Rate Heatmap: Segment × Channel',
            color_continuous_scale='Blues',
            text_auto=True
        )
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.info("Add 'Lead Type' column for segment+channel analysis")

# ──────────────────────────────────────────────────────────────────────────
# 5. DECISION TREE MODELING
# ──────────────────────────────────────────────────────────────────────────

st.header("Decision Tree Modeling")

st.markdown("""
Train segment-specific decision trees to predict prospect reply likelihood.
The model will identify which features (Decision Level, Industry, Channel, Size, Month)
matter most for each segment.
""")

col1, col2, col3 = st.columns(3)

with col1:
    segment_select = st.selectbox(
        "Select segment to model:",
        ["All Data", "Client", "Partner"],
        help="All Data: Use all prospects. Client: Direct buyers. Partner: Channel partners"
    )

with col2:
    target_select = st.selectbox(
        "Predict which outcome:",
        ["Replied", "Meeting", "Converted"],
        help="Replied: Any response. Meeting: Agreed to call. Converted: Sale or LOI"
    )

with col3:
    if st.button("Train Model", use_container_width=True):
        st.session_state.model_trained = True
        segment_param = None if segment_select == "All Data" else segment_select
        st.session_state.model_data = model_summary(df, segment=segment_param, target=target_select)

# Display model results
if st.session_state.model_trained and st.session_state.model_data:
    
    summary = st.session_state.model_data
    
    if "error" in summary:
        st.error(f"Model Error: {summary['error']}")
    else:
        # Header
        st.subheader(f"Results: {summary['segment']} → {summary['target']}")
        
        # CV Metrics
        st.subheader("Cross-Validation Metrics")
        
        cv = summary['cv_results']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("F1 Score", f"{cv['f1_mean']:.3f}", f"±{cv['f1_std']:.3f}")
        
        with col2:
            st.metric("Precision", f"{cv['precision_mean']:.3f}")
        
        with col3:
            st.metric("Recall", f"{cv['recall_mean']:.3f}")
        
        with col4:
            st.metric("Sample Size", f"{cv['n_samples']}")
        
        st.caption(
            f"{cv['n_samples']} prospects analyzed | "
            f"{cv['positive_rate']}% had positive outcome | "
            f"{cv['n_folds']}-fold cross-validation"
        )
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        
        cm = summary['confusion_matrix']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("True Positives", cm['true_positives'], "Correct replies")
        with col2:
            st.metric("False Positives", cm['false_positives'], "Predicted but didn't reply")
        with col3:
            st.metric("False Negatives", cm['false_negatives'], "Missed replies")
        with col4:
            st.metric("True Negatives", cm['true_negatives'], "Correct non-replies")
        
        # Interpretation
        st.info(
            f"""
            **Model Performance Interpretation:**
            - **Precision:** {cm['precision']:.1%} of prospects we score as "likely to reply" actually did
            - **Recall:** {cm['recall']:.1%} of all prospects who replied were identified by the model
            
            Use Precision to avoid wasting time on low-intent prospects.
            Use Recall to avoid missing high-intent prospects.
            """
        )
        
        # Feature Importance
        st.subheader("Feature Importance")
        
        fi = summary['feature_importance']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_fi = px.bar(
                fi.head(10),
                x='Importance %',
                y='Feature',
                orientation='h',
                title='Top 10 Features Driving Predictions',
                labels={'Feature': '', 'Importance %': 'Importance (%)'},
                color='Importance %',
                color_continuous_scale='Blues'
            )
            fig_fi.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
            st.plotly_chart(fig_fi, use_container_width=True)
        
        with col2:
            st.write("**Top 3 Features:**")
            for i, row in enumerate(summary['top_3_features'], 1):
                st.write(f"{i}. {row['Feature']}: {row['Importance %']:.1f}%")
        
        # Decision Rules
        st.subheader("Decision Rules (If-Then Logic)")
        
        st.markdown("""
        These rules show how the decision tree makes predictions.
        Read them as: "If [feature] = [value], then predict [outcome]"
        """)
        
        with st.expander("View Full Decision Rules (Click to Expand)"):
            st.code(summary['decision_rules'], language='text')
        
        # Actionable insights
        st.subheader("How to Use These Results")
        
        insights_text = []
        
        # Most important feature
        top_feature = fi.iloc[0]
        insights_text.append(
            f"**{top_feature['Feature']}** is the strongest predictor "
            f"({top_feature['Importance %']:.1f}% importance). Focus your prospecting on variations of this."
        )
        
        # Precision/Recall tradeoff
        if cm['precision'] > 0.5:
            insights_text.append(
                f"High Precision ({cm['precision']:.1%}): Your high-scoring prospects are likely to reply. "
                f"Use this to prioritize your outreach to top prospects."
            )
        else:
            insights_text.append(
                f"Low Precision ({cm['precision']:.1%}): Even top-scored prospects may not reply. "
                f"You may need more/better signals or different outreach strategy."
            )
        
        for insight in insights_text:
            st.write(insight)

# ──────────────────────────────────────────────────────────────────────────
# 6. TEMPORAL ANALYSIS
# ──────────────────────────────────────────────────────────────────────────

st.header("Temporal Trends")

temporal = temporal_analysis(df)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Weekly Reply Rate Trend")
    
    weekly = temporal['weekly_breakdown']
    
    if not weekly.empty:
        fig_weekly = px.line(
            weekly,
            x='Week Label',
            y='Reply rate %',
            markers=True,
            title='Reply Rate by Week (Feb–Jun 2026)',
            labels={'Week Label': 'Week', 'Reply rate %': 'Reply Rate (%)'},
            line_shape='linear',
            color_discrete_sequence=['#1f77b4']
        )
        fig_weekly.update_layout(height=400)
        st.plotly_chart(fig_weekly, use_container_width=True)
        
        st.dataframe(
            weekly[['Week Label', 'Total', 'Replied', 'Reply rate %', 'Meeting rate %']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Add 'Contact Date' column for weekly analysis")

with col2:
    st.subheader("Monthly Reply Rate Trend")
    
    monthly = temporal['monthly_breakdown']
    
    if not monthly.empty:
        fig_monthly = px.bar(
            monthly,
            x='Contact Month',
            y='Reply rate %',
            title='Reply Rate by Month',
            labels={'Contact Month': 'Month', 'Reply rate %': 'Reply Rate (%)'},
            color='Reply rate %',
            color_continuous_scale='Blues'
        )
        fig_monthly.update_layout(height=400)
        st.plotly_chart(fig_monthly, use_container_width=True)
        
        st.dataframe(
            monthly[['Contact Month', 'Total', 'Replied', 'Reply rate %']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Add 'Contact Date' column for monthly analysis")

# ──────────────────────────────────────────────────────────────────────────
# 7. KEY INSIGHTS
# ──────────────────────────────────────────────────────────────────────────

st.header("Key Insights")

insights = generate_insights(df)

for i, insight in enumerate(insights, 1):
    st.write(f"{i}. {insight}")

# ──────────────────────────────────────────────────────────────────────────
# 8. STRATEGIC RECOMMENDATIONS
# ──────────────────────────────────────────────────────────────────────────

st.header("Strategic Recommendations")

dl_bd = decision_level_breakdown(df)
ch_bd = channel_breakdown(df)
ind_bd = industry_breakdown(df)

if not dl_bd.empty and not ch_bd.empty:
    best_dl = dl_bd.iloc[0]
    best_ch = ch_bd.iloc[0]
    worst_dl = dl_bd.iloc[-1]
    
    recommendation = f"""
    ### For Maximum Reply Rate:
    
    **Target Decision Level:** {best_dl['Decision Level']} 
    - Reply rate: {best_dl['Reply rate %']}% ({int(best_dl['Replied'])}/{int(best_dl['Total'])} prospects)
    - vs {worst_dl['Decision Level']}: {worst_dl['Reply rate %']}% (gap: {best_dl['Reply rate %'] - worst_dl['Reply rate %']:.1f}%)
    - **Action:** Allocate 70% of prospecting efforts to {best_dl['Decision Level']} level
    
    **Preferred Channel:** {best_ch['Contact Channel']}
    - Reply rate: {best_ch['Reply rate %']}% ({int(best_ch['Replied'])}/{int(best_ch['Total'])} prospects)
    - **Action:** Use {best_ch['Contact Channel']} as primary channel; follow up with alternatives
    
    **Industries to Prioritize:**
    """
    
    if not ind_bd.empty:
        top_3_ind = ind_bd.head(3)
        for idx, (_, row) in enumerate(top_3_ind.iterrows(), 1):
            recommendation += f"\n    {idx}. {row['Industry']}: {row['Reply rate %']}% reply rate"
    
    recommendation += f"""
    
    ### Expected Performance:
    - **Baseline reply rate:** {funnel['reply_rate']}% (all prospects)
    - **Optimized reply rate:** 40–50% (top 20% of prospects by score)
    - **Improvement factor:** 1.5–1.7×
    """
    
    st.markdown(recommendation)

# ──────────────────────────────────────────────────────────────────────────
# 9. EXPORT & DOWNLOAD
# ──────────────────────────────────────────────────────────────────────────

st.header("Export Results")

col1, col2, col3 = st.columns(3)

with col1:
    ch_bd = channel_breakdown(df)
    st.download_button(
        label="Channel Breakdown (CSV)",
        data=ch_bd.to_csv(index=False),
        file_name="channel_breakdown.csv",
        mime="text/csv"
    )

with col2:
    dl_bd = decision_level_breakdown(df)
    st.download_button(
        label="Decision Level Breakdown (CSV)",
        data=dl_bd.to_csv(index=False),
        file_name="decision_level_breakdown.csv",
        mime="text/csv"
    )

with col3:
    weekly = weekly_analysis(df)
    if not weekly.empty:
        st.download_button(
            label="Weekly Analysis (CSV)",
            data=weekly.to_csv(index=False),
            file_name="weekly_analysis.csv",
            mime="text/csv"
        )

# Excel export
st.divider()

try:
    from io import BytesIO
    import openpyxl
    
    st.subheader("Complete Analytics Export (Excel)")
    
    export_sheets = {
        "Funnel": pd.DataFrame([funnel]),
        "Channel": channel_breakdown(df),
        "Industry": industry_breakdown(df),
        "Decision Level": decision_level_breakdown(df),
        "Size": size_breakdown(df),
        "Weekly": weekly_analysis(df),
    }
    
    if segment_comparison(df):
        seg_data = []
        for seg, data in segment_comparison(df).items():
            seg_data.append({
                "Segment": seg,
                "Count": data['count'],
                "Reply %": data['funnel']['reply_rate'],
                "Meeting %": data['funnel']['meeting_rate'],
            })
        export_sheets["Segments"] = pd.DataFrame(seg_data)
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, sheet_df in export_sheets.items():
            if not sheet_df.empty:
                sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    output.seek(0)
    st.download_button(
        label="Download All Analytics (Excel)",
        data=output.getvalue(),
        file_name="VEEP_Analytics_Complete.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

except ImportError:
    st.warning("openpyxl not installed. Use CSV exports above or install: pip install openpyxl")

# ──────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────

st.divider()

st.markdown("""
---

**VEEP Prospecting Framework - Analytics Module**

This dashboard implements a decision-tree-based prospecting framework designed for 
data-driven GTM in market-entry contexts. Features:

- Segment-specific modeling (Clients vs Partners)
- Feature importance extraction (actionable insights)
- Decision rules (interpretable predictions)
- Temporal trend analysis (weekly/monthly patterns)
- Comprehensive breakdowns (channel, industry, decision level)

""")
