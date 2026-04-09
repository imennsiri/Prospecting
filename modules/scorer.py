"""
Fit scoring engine
Combines news, job, and website signals into a 0-100 fit score.
Designed to produce meaningful score distribution across real French companies.
"""

def score_prospect(
    company_name:  str,
    news_signals:  list[dict],
    job_signals:   list[dict]
) -> tuple[int, list[str]]:
    """
    Returns (score: int, reasons: list[str])
    Score is 0-100. Reasons explain what drove the score.

    Scoring breakdown:
    - Base score:        10 pts (every company starts here)
    - Job signals:       max 25 pts
    - News signals:      max 25 pts
    - Website signals:   max 25 pts
    - Coverage bonus:    max 15 pts
    Total possible:      100 pts
    """
    score   = 10  # base score — company exists and was found
    reasons = []

    # ── Job signals (max 25 pts) ──────────────────────────────────────────────
    strong_jobs = [j for j in job_signals if j["signal_strength"] == 2]
    medium_jobs = [j for j in job_signals if j["signal_strength"] == 1]

    if strong_jobs:
        pts = min(len(strong_jobs) * 15, 25)
        score += pts
        titles = ", ".join(j["title"] for j in strong_jobs[:2])
        reasons.append(f"Hiring for people/HR roles: {titles}")
    elif medium_jobs:
        pts = min(len(medium_jobs) * 8, 15)
        score += pts
        reasons.append(f"{len(medium_jobs)} adjacent HR-related posting(s) found")
    else:
        reasons.append("No HR/People job postings found")

    # ── News signals (max 25 pts) ─────────────────────────────────────────────
    high_relevance = [n for n in news_signals if n["relevance"] >= 2]
    low_relevance  = [n for n in news_signals if n["relevance"] == 1]
    any_news       = len(news_signals) > 0

    if high_relevance:
        pts = min(len(high_relevance) * 15, 25)
        score += pts
        reasons.append(f"Strong growth signal: \"{high_relevance[0]['title'][:60]}...\"")
    elif low_relevance:
        pts = min(len(low_relevance) * 8, 15)
        score += pts
        reasons.append("Some growth activity detected in news")
    elif any_news:
        score += 5  # company appears in news at all — mild positive signal
        reasons.append("Company present in recent news (no strong growth keywords)")
    else:
        reasons.append("No recent news found")

    # ── Website signals (max 25 pts) ──────────────────────────────────────────
    from modules.job_scraper import scrape_website_signals

    website_data = scrape_website_signals(company_name)
    if website_data["boost"] > 0:
        pts = min(website_data["boost"], 25)
        score += pts
        found_preview = ", ".join(website_data["found"][:3])
        reasons.append(f"Website mentions VEEP-relevant keywords: {found_preview}")
    elif website_data["status"] == "ok":
        score += 5  # website accessible even if no keywords — mild positive
        reasons.append("Website accessible but no VEEP-relevant keywords detected")
    else:
        reasons.append("Could not access company website")

    # ── Coverage bonus (max 15 pts) ───────────────────────────────────────────
    signals_found = sum([
        1 if job_signals  else 0,
        1 if news_signals else 0,
        1 if website_data.get("status") == "ok" else 0,
    ])
    if signals_found == 3:
        score += 15
        reasons.append("Strong data coverage across all three sources")
    elif signals_found == 2:
        score += 10
        reasons.append("Partial data coverage (2 of 3 sources)")

    # Cap at 100
    score = min(score, 100)

    return score, reasons