import streamlit as st
import pandas as pd
from modules.news_fetcher import fetch_signals
from modules.job_scraper import fetch_job_signals
from modules.scorer import score_prospect
from modules.message_generator import generate_message
from modules.tracker import load_results, save_result
from modules.analytics import (
    load_and_validate, funnel_stats, channel_breakdown,
    industry_breakdown, decision_level_breakdown, size_breakdown,
    logistic_regression_analysis, temporal_analysis, generate_insights, weekly_analysis
)

st.set_page_config(
    page_title="VEEP Prospect Tool",
    layout="wide"
)

st.title("VEEP Prospect Intelligence Tool")
st.caption("Automated research, scoring & outreach generation for French B2B prospects")
import streamlit as st

# ── Simple password gate ───────────────────────────────────────────────────
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("VEEP Prospect Tool")
    st.markdown("#### Please enter the access password")
    password = st.text_input("Password", type="password", placeholder="Enter password...")

    if st.button("Login"):
        if password == st.secrets["APP_PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")

    return False

if not check_password():
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    score_threshold = st.slider("Minimum fit score to show", 0, 100, 20)
    auto_generate = st.toggle("Auto-generate messages", value=True)
    st.divider()
    st.markdown("**How scoring works**")
    st.caption("Scores are based on HR hiring signals, company growth news, and size fit for VEEP.")

# ── Tab layout ────────────────────────────────────────────────────────────────
tab_prospect, tab_analytics = st.tabs([
    "Prospect a company",
    "Analytics"
])

# ── TAB 1: Prospect a company ─────────────────────────────────────────────────
with tab_prospect:
    st.subheader("Research a new prospect")

    col1, col2 = st.columns([2, 1])
    with col1:
        company_input = st.text_area(
            "Enter company names (one per line)",
            placeholder="Exemple:\nMonoprix\nBiat\n",
            height=150
        )
    with col2:
        st.markdown("**Tips**")
        st.caption("Use the exact company name as it appears on their website or LinkedIn.")
        st.caption("You can paste up to 20 companies at once.")

    run_btn = st.button("Research & Score", type="primary", use_container_width=True)

    if run_btn and company_input.strip():
        companies = [c.strip() for c in company_input.strip().split("\n") if c.strip()]
        results = []

        progress = st.progress(0, text="Starting research...")

        for i, company in enumerate(companies):
            progress.progress((i) / len(companies), text=f"Researching {company}...")

            with st.spinner(f"Fetching signals for {company}..."):
                news_signals    = fetch_signals(company)
                score, reasons  = score_prospect(company, news_signals)
                message         = generate_message(company, news_signals, score) if auto_generate else ""

            results.append({
                "company":      company,
                "score":        score,
                "reasons":      reasons,
                "news":         news_signals,
                "message":      message,
            })

        progress.progress(1.0, text="Done!")
        st.session_state["last_results"] = results

    # Display results from this run
    if "last_results" in st.session_state:
        st.divider()
        filtered = [r for r in st.session_state["last_results"] if r["score"] >= score_threshold]
        st.markdown(f"**{len(filtered)} prospects** above score threshold ({score_threshold})")

        for r in sorted(filtered, key=lambda x: x["score"], reverse=True):
            score_color = "🟢" if r["score"] >= 70 else "🟡" if r["score"] >= 40 else "🔴"
            with st.expander(f"{score_color} **{r['company']}** — Score: {r['score']}/100"):

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Why this score**")
                    for reason in r["reasons"]:
                        st.caption(f"• {reason}")

                    st.markdown("**News signals**")
                    if r["news"]:
                        for item in r["news"][:3]:
                            st.caption(f"📰 [{item['title']}]({item['url']})")
                    else:
                        st.caption("No recent news found.")

                with col_b:
                    st.markdown("**Outreach message**")
                    message_text = st.text_area(
                        "Edit before sending",
                        value=r["message"],
                        height=200,
                        key=f"msg_{r['company']}"
                    )

                col_sent, col_save, col_skip = st.columns(3)
                with col_sent:
                    if st.button("✅ Mark as sent", key=f"sent_{r['company']}"):
                        save_result(r['company'], r['score'], "sent", message_text)
                        st.success("Saved!")
                with col_save:
                    if st.button("🔖 Save lead", key=f"save_{r['company']}"):
                        save_result(r['company'], r['score'], "saved", "")
                        st.success("Lead saved!")
                with col_skip:
                    if st.button("⏭ Skip", key=f"skip_{r['company']}"):
                        save_result(r['company'], r['score'], "skipped", "")
                        st.info("Skipped.")

# ── TAB 4: Analytics ──────────────────────────────────────────────────────────
with tab_analytics:
    st.subheader("Prospect analytics")
    st.caption("Upload your existing prospect spreadsheet to analyze conversion patterns.")
    uploaded_file = st.file_uploader(
      "Upload your prospects CSV or Excel file",
        type=["csv", "xlsx"],
        help="Must include columns: Company, Industry, Size, Contact Channel, Decision Level, Contact Month, Replied, Meeting, Converted"
    )

    if uploaded_file:
        # Load file
        if uploaded_file.name.endswith(".csv"):
            raw_df = pd.read_csv(uploaded_file)
        else:
           raw_df = pd.read_excel(uploaded_file)

        df_clean, warnings = load_and_validate(raw_df)

        if warnings:
            for w in warnings:
                st.warning(w)

        st.success(f"Loaded {len(df_clean)} prospects.")
        st.session_state["analytics_df"] = df_clean

    if "analytics_df" in st.session_state:
        adf = st.session_state["analytics_df"]

        # ── Key insights ──────────────────────────────────────────────────────
        st.divider()
        st.markdown("### Key insights")
        insights = generate_insights(adf)
        for insight in insights:
           st.markdown(f"• {insight}")

        # ── Conversion funnel ─────────────────────────────────────────────────
        st.divider()
        st.markdown("### Conversion funnel")
        funnel = funnel_stats(adf)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Prospects contacted", funnel["total"])
        c2.metric("Replied",    funnel["replied"],   f"{funnel['reply_rate']}% of contacted")
        c3.metric("Meeting",    funnel["meeting"],   f"{funnel['meeting_rate']}% of replies")
        c4.metric("Converted",  funnel["converted"], f"{funnel['conversion_rate']}% of meetings")

        # ── Breakdown tables ──────────────────────────────────────────────────
        st.divider()
        st.markdown("### Breakdown analyses")

        breakdown_tab1, breakdown_tab2, breakdown_tab3, breakdown_tab4 = st.tabs([
            "By channel", "By industry", "By decision level", "By company size"
        ])

        with breakdown_tab1:
            st.dataframe(channel_breakdown(adf), use_container_width=True, hide_index=True)

        with breakdown_tab2:
            st.dataframe(industry_breakdown(adf), use_container_width=True, hide_index=True)

        with breakdown_tab3:
            st.dataframe(decision_level_breakdown(adf), use_container_width=True, hide_index=True)

        with breakdown_tab4:
            st.dataframe(size_breakdown(adf), use_container_width=True, hide_index=True)

        # ── Temporal analysis ─────────────────────────────────────────────────
        st.divider()
        st.markdown("### Temporal analysis")
        temporal = temporal_analysis(adf)
        st.markdown(temporal["insight"])

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("**Volume per month**")
            st.dataframe(temporal["volume"], use_container_width=True, hide_index=True)
        with col_t2:
            st.markdown("**Reply & meeting rates per month**")
            st.dataframe(
                temporal["monthly_breakdown"][["Contact Month", "Total", "Reply rate %", "Meeting rate %"]],
                use_container_width=True, hide_index=True
            )
        if "Contact Date" in adf.columns:
            st.divider()
            st.markdown("### Weekly trend analysis")
        weekly = weekly_analysis(adf)
        if not weekly.empty:
            st.dataframe(weekly, use_container_width=True, hide_index=True)
            st.caption("This is your time series — reply rate evolution week by week.")
            # ── Logistic regression ───────────────────────────────────────────────
            st.divider()
            st.markdown("### Logistic regression : what predicts a reply?")
            st.caption("This model identifies which factors are most associated with a prospect replying.")

            reg = logistic_regression_analysis(adf)

            if "error" in reg:
                st.warning(reg["error"])
            else:
                col_r1, col_r2 = st.columns([1, 1])

                with col_r1:
                    st.markdown("**Model accuracy**")
                    st.metric("Accuracy", f"{reg['accuracy']}%",
                        help="How often the model correctly predicts reply/no reply on training data.")
                    st.caption(f"Trained on {reg['n_samples']} prospects, {reg['n_replied']} replies.")

                    st.markdown("**Coefficients**")
                    st.dataframe(reg["coefficients"], use_container_width=True, hide_index=True)

                with col_r2:
                    st.markdown("**Interpretation**")
                    for insight in reg["interpretation"]:
                        st.markdown(f"• {insight}")

                    st.divider()
                    st.markdown("**What this means for your outreach strategy**")
                    top = reg["top_feature"]
                    st.info(
                        f"Focus on optimizing **{top}** first — "
                        f"it has the strongest influence on whether a prospect replies."
                    )

        # ── Export ────────────────────────────────────────────────────────────
        st.divider()
        csv_export = adf.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export cleaned dataset as CSV",
            csv_export,
            "veep_prospects_clean.csv",
            "text/csv"
        )
