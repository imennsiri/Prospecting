"""
News & signals fetcher
Fetches recent news about a French company using Google News RSS.
Returns a list of signal dicts: {title, url, date, relevance}
"""

import feedparser
import urllib.parse
from datetime import datetime

GROWTH_KEYWORDS = [
    "lève", "levée de fonds", "financement", "série", "expansion",
    "recrute", "recrutement", "croissance", "ouverture", "nouveau siège",
    "fusion", "acquisition", "partenariat", "lancement", "formation", "training"
]

def fetch_signals(company_name: str) -> list[dict]:
    """
    Fetches recent news for a French company via Google News RSS.
    Returns up to 5 relevant signals.
    """
    query = urllib.parse.quote(f"{company_name} France")
    url = f"https://news.google.com/rss/search?q={query}&hl=fr&gl=FR&ceid=FR:fr"

    try:
        feed = feedparser.parse(url)
        signals = []

        for entry in feed.entries[:10]:
            title = entry.get("title", "")
            link  = entry.get("link", "")
            date  = entry.get("published", "")

            relevance = _score_relevance(title)

            signals.append({
                "title":     title,
                "url":       link,
                "date":      date,
                "relevance": relevance
            })

        # Sort by relevance (growth signals first)
        signals.sort(key=lambda x: x["relevance"], reverse=True)
        return signals[:5]

    except Exception as e:
        print(f"[news_fetcher] Error fetching news for {company_name}: {e}")
        return []


def _score_relevance(title: str) -> int:
    """
    Scores a news title based on growth/hiring keywords.
    Higher = more relevant to VEEP prospecting timing.
    """
    title_lower = title.lower()
    score = 0
    for keyword in GROWTH_KEYWORDS:
        if keyword in title_lower:
            score += 1
    return score
