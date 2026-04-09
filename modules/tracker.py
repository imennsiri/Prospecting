"""
Results tracker
Saves and loads prospect outreach results to a local CSV.
This is your feedback loop — the data here will later be used
to tune the scoring engine.
"""

import pandas as pd
import os
from datetime import datetime

DATA_PATH = "data/results.csv"

COLUMNS = ["company", "score", "status", "message", "date_added", "date_updated"]

def load_results() -> pd.DataFrame:
    """Loads the results CSV. Returns empty DataFrame if file doesn't exist."""
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame(columns=COLUMNS)


def save_result(company: str, score: int, status: str, message: str = ""):
    """
    Saves or updates a prospect result.
    If company already exists, updates status and date.
    """
    df = load_results()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    if company in df["company"].values:
        df.loc[df["company"] == company, "status"]       = status
        df.loc[df["company"] == company, "date_updated"] = now
        if message:
            df.loc[df["company"] == company, "message"]  = message
    else:
        new_row = pd.DataFrame([{
            "company":      company,
            "score":        score,
            "status":       status,
            "message":      message,
            "date_added":   now,
            "date_updated": now
        }])
        df = pd.concat([df, new_row], ignore_index=True)

    os.makedirs("data", exist_ok=True)
    df.to_csv(DATA_PATH, index=False)


def get_conversion_stats() -> dict:
    """
    Returns basic conversion stats for the results tracker tab.
    Useful later for tuning the scoring model.
    """
    df = load_results()
    if df.empty:
        return {}

    return {
        "total":        len(df),
        "sent":         len(df[df["status"] == "sent"]),
        "replied":      len(df[df["status"] == "replied"]),
        "demo_booked":  len(df[df["status"] == "demo_booked"]),
        "avg_score_replied": df[df["status"] == "replied"]["score"].mean() if len(df[df["status"] == "replied"]) > 0 else 0
    }
