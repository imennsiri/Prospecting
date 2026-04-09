import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")


# ── Data loading & validation ─────────────────────────────────────────────────

REQUIRED_COLUMNS = [
    "Company", "City", "Industry", "Size", "Lead Source",
    "Prospect Title", "Decision Level", "Contacted",
    "Contact Month","Contact Date", "Contact Channel", "Replied", "Meeting", "Converted"
]

def load_and_validate(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Validates uploaded dataframe and normalizes column names.
    Returns (cleaned_df, list_of_warnings)
    """
    warnings_list = []

    # Normalize column names: strip spaces, title case
    df.columns = df.columns.str.strip()

    # Check for required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        warnings_list.append(f"Missing columns: {', '.join(missing)}")

    # Ensure binary columns are numeric
    for col in ["Replied", "Meeting", "Converted"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        if "Contact Date" in df.columns:
            df["Contact Date"] = pd.to_datetime(df["Contact Date"], dayfirst=True, errors="coerce")
            df["Week"] = df["Contact Date"].dt.isocalendar().week.astype(str)
            df["Week Label"] = df["Contact Date"].dt.strftime("W%V %b")

    # Normalize channel values
    if "Contact Channel" in df.columns:
        df["Contact Channel"] = df["Contact Channel"].str.strip().str.title()

    # Keep Size as-is (string dropdown values like "200-500", "501-1000", "1001-1500")
    if "Size" in df.columns:
        df["Size"] = df["Size"].astype(str).str.strip()

    return df, warnings_list


def size_bucket(size: str) -> str:
    """Pass-through since Size is already a labeled range from the dropdown."""
    if pd.isna(size) or str(size) in ["nan", "", "None"]:
        return "Unknown"
    return str(size)


# ── Funnel stats ──────────────────────────────────────────────────────────────

def funnel_stats(df: pd.DataFrame) -> dict:
    """Returns conversion funnel numbers."""
    total     = len(df)
    replied   = df["Replied"].sum()
    meeting   = df["Meeting"].sum()
    converted = df["Converted"].sum()

    return {
        "total":            int(total),
        "replied":          int(replied),
        "meeting":          int(meeting),
        "converted":        int(converted),
        "reply_rate":       round(replied   / total   * 100, 1) if total   else 0,
        "meeting_rate":     round(meeting   / replied * 100, 1) if replied else 0,
        "conversion_rate":  round(converted / meeting * 100, 1) if meeting else 0,
    }


# ── Breakdown analyses ────────────────────────────────────────────────────────

def breakdown_by(df: pd.DataFrame, column: str, target: str = "Replied") -> pd.DataFrame:
    """
    Returns reply/meeting/converted rates grouped by a given column.
    """
    group = df.groupby(column).agg(
        Total       = (target, "count"),
        Replied     = ("Replied",   "sum"),
        Meeting     = ("Meeting",   "sum"),
        Converted   = ("Converted", "sum"),
    ).reset_index()

    group["Reply rate %"]   = (group["Replied"] / group["Total"]   * 100).round(1)
    group["Meeting rate %"] = (group["Meeting"] / group["Replied"].replace(0, float("nan")) * 100).round(1).fillna(0)
    group["Convert rate %"] = (group["Converted"] / group["Meeting"].replace(0, float("nan")) * 100).round(1).fillna(0)

    return group.sort_values("Reply rate %", ascending=False)


def channel_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    return breakdown_by(df, "Contact Channel")


def industry_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    return breakdown_by(df, "Industry")


def decision_level_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    return breakdown_by(df, "Decision Level")


def monthly_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Reply and meeting rates per contact month."""
    return breakdown_by(df, "Contact Month")

def daily_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Reply and meeting rates per contact date."""
    return breakdown_by(df, "Contact Date")

def size_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Adds size bucket column and groups by it."""
    df = df.copy()
    df["Size bucket"] = df["Size"].apply(size_bucket)
    return breakdown_by(df, "Size bucket")


# ── Logistic regression ───────────────────────────────────────────────────────

def logistic_regression_analysis(df: pd.DataFrame) -> dict:
    """
    Runs logistic regression to predict Reply (1/0) from:
    - Contact Channel (encoded)
    - Decision Level (encoded)
    - Industry (encoded)
    - Size bucket (encoded)
    - Contact Month (encoded)
    - Contact Date (encoded)

    Returns coefficients, model accuracy, and interpretation.
    """
    df = df.copy()
    df["Size bucket"] = df["Size"].apply(size_bucket)

    feature_cols = ["Contact Channel", "Decision Level", "Industry",
                    "Size bucket", "Contact Month", "Contact Date"]
    target_col   = "Replied"

    # Drop rows with missing values in features or target
    subset = df[feature_cols + [target_col]].dropna()

    if len(subset) < 10:
        return {"error": "Not enough data for regression (need at least 10 complete rows)."}

    if subset[target_col].nunique() < 2:
        return {"error": "Target variable has only one class — need both replies and non-replies."}

    # Encode categorical features
    encoders = {}
    X = pd.DataFrame()
    for col in feature_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(subset[col].astype(str))
        encoders[col] = le

    y = subset[target_col].astype(int)

    # Fit model
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X, y)

    # Build coefficients table
    coef_df = pd.DataFrame({
        "Feature":     feature_cols,
        "Coefficient": model.coef_[0].round(3),
    })
    coef_df["Impact"] = coef_df["Coefficient"].apply(
        lambda c: "↑ Positive" if c > 0.1 else ("↓ Negative" if c < -0.1 else "→ Neutral")
    )
    coef_df = coef_df.sort_values("Coefficient", ascending=False)

    # Accuracy
    preds    = model.predict(X)
    accuracy = round((preds == y).mean() * 100, 1)

    # Dominant positive feature
    top_feature = coef_df.iloc[0]["Feature"]
    top_coef    = coef_df.iloc[0]["Coefficient"]

    interpretation = _interpret_coefficients(coef_df)

    return {
        "coefficients":    coef_df,
        "accuracy":        accuracy,
        "n_samples":       len(subset),
        "n_replied":       int(y.sum()),
        "top_feature":     top_feature,
        "top_coef":        top_coef,
        "interpretation":  interpretation,
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
    }


def _interpret_coefficients(coef_df: pd.DataFrame) -> list[str]:
    """Generates plain-language interpretations of regression results."""
    insights = []
    for _, row in coef_df.iterrows():
        c = row["Coefficient"]
        f = row["Feature"]
        if c > 0.3:
            insights.append(f"**{f}** is the strongest positive predictor of a reply.")
        elif c > 0.1:
            insights.append(f"**{f}** has a moderate positive effect on reply likelihood.")
        elif c < -0.3:
            insights.append(f"**{f}** is negatively associated with replies — worth investigating.")
        elif c < -0.1:
            insights.append(f"**{f}** slightly reduces reply likelihood.")
    return insights


# ── Temporal pattern analysis ─────────────────────────────────────────────────

def temporal_analysis(df: pd.DataFrame) -> dict:
    """
    Analyzes outreach patterns over Contact Month.

    """
    monthly = monthly_breakdown(df)

    # Volume trend
    volume = df.groupby("Contact Month").size().reset_index(name="Prospects contacted")

    # Best month for replies
    best_month = monthly.loc[monthly["Reply rate %"].idxmax(), "Contact Month"] \
        if not monthly.empty else "N/A"

    return {
        "monthly_breakdown": monthly,
        "volume":            volume,
        "best_month":        best_month,
        "insight": (
            f"Most effective outreach month: **{best_month}** "
            f"based on reply rate."
        )
    }


# ── Key insights summary ──────────────────────────────────────────────────────

def generate_insights(df: pd.DataFrame) -> list[str]:
    """
    Generates a list of the top data-driven insights from the dataset.
    """
    insights = []
    funnel = funnel_stats(df)

    insights.append(
        f"Overall reply rate is **{funnel['reply_rate']}%** "
        f"({funnel['replied']}/{funnel['total']} prospects)."
    )

    # Best channel
    ch = channel_breakdown(df)
    if not ch.empty:
        best_ch = ch.iloc[0]
        insights.append(
            f"**{best_ch['Contact Channel']}** is the most effective channel "
            f"with a {best_ch['Reply rate %']}% reply rate."
        )

    # Best industry
    ind = industry_breakdown(df)
    if not ind.empty:
        best_ind = ind.iloc[0]
        insights.append(
            f"**{best_ind['Industry']}** is the most responsive industry "
            f"({best_ind['Reply rate %']}% reply rate)."
        )

    # Decision level
    dl = decision_level_breakdown(df)
    if not dl.empty:
        best_dl = dl.iloc[0]
        insights.append(
            f"Contacting **{best_dl['Decision Level']}** level yields "
            f"the best reply rate ({best_dl['Reply rate %']}%)."
        )

    # Size
    sz = size_breakdown(df)
    if not sz.empty:
        best_sz = sz.iloc[0]
        insights.append(
            f"Companies sized **{best_sz['Size bucket']}** employees "
            f"respond best ({best_sz['Reply rate %']}% reply rate)."
        )

    return insights
def weekly_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups outreach by week and computes reply rate per week.
    
    """
    if "Week Label" not in df.columns:
        return pd.DataFrame()

    group = df.groupby("Week Label").agg(
        Total    = ("Replied", "count"),
        Replied  = ("Replied", "sum"),
        Meeting  = ("Meeting", "sum"),
    ).reset_index()

    group["Reply rate %"]   = (group["Replied"] / group["Total"] * 100).round(1)
    group["Meeting rate %"] = (group["Meeting"] / group["Replied"].replace(0, float("nan")) * 100).round(1).fillna(0)

    return group
