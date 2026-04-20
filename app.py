import streamlit as st
from auth import check_auth
check_auth()
import pandas as pd
from modules.news_fetcher import fetch_signals
from modules.job_scraper import fetch_job_signals
from modules.scorer import score_prospect
from modules.message_generator import generate_message

st.set_page_config(
    page_title="VEEP Prospect Tool",
    layout="wide"
)

st.title("VEEP Prospect Intelligence Tool")
st.caption("Automated research, scoring & outreach generation for French B2B prospects")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    score_threshold = st.slider("Minimum fit score to show", 0, 100, 20)
    auto_generate = st.toggle("Auto-generate messages", value=True)
    st.divider()
    st.markdown("**How scoring works**")
    st.caption("Scores are based on HR hiring signals, company growth news, and size fit for VEEP.")
    st.sidebar.success("Navigate using the menu above")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
    st.rerun()

st.title("VEEP Prospect Intelligence Tool")
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

