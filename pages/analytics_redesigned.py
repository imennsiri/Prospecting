import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import warnings
warnings.filterwarnings("ignore")


# ── Data loading & validation ─────────────────────────────────────────────────

REQUIRED_COLUMNS = [
    "Lead Type", "Company", "Industry", "Size", "Lead Source",
    "Prospect Title", "Decision Level", "Contacted",
    "Contact Date", "Contact Channel", "Replied", "Meeting", "Converted"
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
        df["Contact Month"] = df["Contact Date"].dt.month
        df["Contact Week"] = df["Contact Date"].dt.isocalendar().week.astype(str)
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


def segment_funnel(df: pd.DataFrame, segment: str) -> dict:
    """
    Returns conversion funnel for a specific Lead Type segment (Client / Partner).
    """
    df_seg = df[df["Lead Type"] == segment]
    return {
        "segment": segment,
        "stats": funnel_stats(df_seg),
        "count": len(df_seg)
    }


# ── Breakdown analyses ────────────────────────────────────────────────────────

def breakdown_by(df: pd.DataFrame, column: str, target: str = "Replied") -> pd.DataFrame:
    """
    Returns reply/meeting/converted rates grouped by a given column.
    """
    group = df.groupby(column, dropna=False).agg(
        Total       = (target, "count"),
        Replied     = ("Replied",   "sum"),
        Meeting     = ("Meeting",   "sum"),
        Converted   = ("Converted", "sum"),
    ).reset_index()

    group["Reply rate %"]   = (group["Replied"] / group["Total"]   * 100).round(1)
    group["Meeting rate %"] = (group["Meeting"] / group["Replied"].replace(0, np.nan) * 100).round(1)
    group["Convert rate %"] = (group["Converted"] / group["Meeting"].replace(0, np.nan) * 100).round(1)

    return group.sort_values("Reply rate %", ascending=False)


def channel_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Breakdown by Contact Channel."""
    return breakdown_by(df, "Contact Channel")


def industry_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Breakdown by Industry."""
    return breakdown_by(df, "Industry")


def decision_level_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Breakdown by Decision Level."""
    return breakdown_by(df, "Decision Level")


def monthly_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Reply and meeting rates per contact month."""
    if "Contact Month" not in df.columns:
        return pd.DataFrame()
    return breakdown_by(df, "Contact Month")


def size_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Adds size bucket column and groups by it."""
    df = df.copy()
    df["Size bucket"] = df["Size"].apply(size_bucket)
    return breakdown_by(df, "Size bucket")


def segment_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Breakdown by Lead Type + Contact Channel (segment-specific insights).
    Shows how different channels perform per segment.
    """
    if "Lead Type" not in df.columns:
        return pd.DataFrame()
    
    group = df.groupby(["Lead Type", "Contact Channel"], dropna=False).agg(
        Total    = ("Replied", "count"),
        Replied  = ("Replied", "sum"),
        Meeting  = ("Meeting", "sum"),
        Converted = ("Converted", "sum"),
    ).reset_index()

    group["Reply rate %"]   = (group["Replied"] / group["Total"]   * 100).round(1)
    group["Meeting rate %"] = (group["Meeting"] / group["Replied"].replace(0, np.nan) * 100).round(1)
    
    return group.sort_values(["Lead Type", "Reply rate %"], ascending=[True, False])


# ── DECISION TREE MODELING ────────────────────────────────────────────────────

def prepare_tree_data(df: pd.DataFrame, segment: str = None, target: str = "Replied") -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepares feature matrix and target for decision tree training.
    
    Args:
        df: Input dataframe
        segment: 'Client', 'Partner', or None (all data)
        target: 'Replied', 'Meeting', or 'Converted'
    
    Returns:
        (X, y) tuple ready for modeling
    """
    df_work = df.copy()
    
    # Filter by segment if specified
    if segment and "Lead Type" in df_work.columns:
        df_work = df_work[df_work["Lead Type"] == segment]
    
    # Select features for modeling
    feature_cols = [
        "Decision Level",
        "Industry",
        "Size",
        "Contact Channel",
        "Contact Month"
    ]
    
    # Check availability
    available_features = [col for col in feature_cols if col in df_work.columns]
    
    # Drop rows with missing target or features
    subset = df_work[available_features + [target]].dropna()
    
    if len(subset) < 8:
        return None, None
    
    X = subset[available_features]
    y = subset[target].astype(int)
    
    return X, y


def build_decision_tree_pipeline(X: pd.DataFrame, y: pd.Series, max_depth: int = 5, 
                                  min_samples_leaf: int = 4, min_samples_split: int = 8) -> Pipeline:
    """
    Builds a preprocessing + decision tree pipeline.
    
    Args:
        X: Feature matrix
        y: Target vector
        max_depth: Tree max depth (controls complexity)
        min_samples_leaf: Minimum samples required at leaf node
        min_samples_split: Minimum samples required to split internal node
    
    Returns:
        Fitted sklearn Pipeline
    """
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Preprocessing: OneHotEncode categoricals, keep numerics as-is
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
             categorical_features),
            ('num', 'passthrough', numeric_features)
        ],
        remainder='passthrough'
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('tree', DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            class_weight='balanced',
            random_state=42
        ))
    ])
    
    pipeline.fit(X, y)
    return pipeline


def cross_validate_tree(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
    """
    Performs stratified K-fold cross-validation on decision tree.
    
    Returns metrics including mean F1, precision, recall across folds.
    """
    if len(X) < n_splits + 5:
        return {"error": f"Not enough data for {n_splits}-fold CV (need at least {n_splits + 5} samples)"}
    
    if y.nunique() < 2:
        return {"error": "Target has only one class — need both 0 and 1 outcomes"}
    
    # Prepare pipeline
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
             categorical_features),
            ('num', 'passthrough', numeric_features)
        ],
        remainder='passthrough'
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('tree', DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=4,
            min_samples_split=8,
            class_weight='balanced',
            random_state=42
        ))
    ])
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    f1_scores = cross_val_score(pipeline, X, y, cv=skf, scoring='f1_weighted')
    precision_scores = cross_val_score(pipeline, X, y, cv=skf, scoring='precision_weighted')
    recall_scores = cross_val_score(pipeline, X, y, cv=skf, scoring='recall_weighted')
    
    return {
        "n_samples": len(X),
        "n_folds": n_splits,
        "f1_mean": round(f1_scores.mean(), 3),
        "f1_std": round(f1_scores.std(), 3),
        "f1_scores": [round(s, 3) for s in f1_scores],
        "precision_mean": round(precision_scores.mean(), 3),
        "recall_mean": round(recall_scores.mean(), 3),
        "positive_rate": round(y.sum() / len(y) * 100, 1),
    }


def extract_feature_importance(pipeline: Pipeline, feature_names: list) -> pd.DataFrame:
    """
    Extracts feature importance from trained decision tree.
    
    Args:
        pipeline: Fitted sklearn Pipeline
        feature_names: Original feature names before preprocessing
    
    Returns:
        DataFrame sorted by importance (descending)
    """
    tree = pipeline.named_steps['tree']
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Get transformed feature names
    try:
        transformed_names = preprocessor.get_feature_names_out()
    except:
        transformed_names = [f"Feature_{i}" for i in range(tree.n_features_in_)]
    
    importance_df = pd.DataFrame({
        'Feature': transformed_names,
        'Importance': tree.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    importance_df['Importance %'] = (importance_df['Importance'] * 100).round(1)
    
    return importance_df


def extract_decision_rules(pipeline: Pipeline, feature_names: list = None, max_depth: int = 3) -> str:
    """
    Extracts human-readable decision rules from tree.
    
    Args:
        pipeline: Fitted sklearn Pipeline
        feature_names: Original feature names (optional)
        max_depth: Limit rule depth for readability
    
    Returns:
        String representation of decision tree rules
    """
    tree = pipeline.named_steps['tree']
    preprocessor = pipeline.named_steps['preprocessor']
    
    try:
        feature_names_out = preprocessor.get_feature_names_out()
    except:
        feature_names_out = [f"Feature_{i}" for i in range(tree.n_features_in_)]
    
    rules = export_text(tree, feature_names=feature_names_out, max_depth=max_depth)
    return rules


def score_prospects(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Generates probability scores (0-100) for all prospects.
    
    Returns:
        Array of scores 0-100 (probability of positive outcome)
    """
    proba = pipeline.predict_proba(X)[:, 1]  # Probability of class 1
    return (proba * 100).astype(int)


def model_summary(df: pd.DataFrame, segment: str = None, target: str = "Replied") -> dict:
    """
    Complete decision tree modeling summary for a segment.
    
    Returns:
        Dictionary with CV scores, feature importance, rules, and model stats
    """
    X, y = prepare_tree_data(df, segment=segment, target=target)
    
    if X is None:
        return {"error": f"Insufficient data for {segment or 'all'} segment (need ≥8 samples)"}
    
    # Cross-validation
    cv_results = cross_validate_tree(X, y)
    if "error" in cv_results:
        return cv_results
    
    # Build and fit pipeline
    pipeline = build_decision_tree_pipeline(X, y)
    
    # Feature importance
    feature_importance = extract_feature_importance(pipeline, X.columns.tolist())
    
    # Decision rules
    rules = extract_decision_rules(pipeline, X.columns.tolist(), max_depth=3)
    
    # Predictions for confusion matrix
    y_pred = pipeline.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    
    precision = round(tp / (tp + fp), 3) if (tp + fp) > 0 else 0
    recall = round(tp / (tp + fn), 3) if (tp + fn) > 0 else 0
    
    return {
        "segment": segment or "All Data",
        "target": target,
        "cv_results": cv_results,
        "feature_importance": feature_importance,
        "decision_rules": rules,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "precision": precision,
            "recall": recall,
        },
        "top_3_features": feature_importance.head(3)[['Feature', 'Importance %']].to_dict('records'),
    }


# ── Temporal analysis ────────────────────────────────────────────────────────

def weekly_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups outreach by week and computes reply rate per week.
    """
    if "Week Label" not in df.columns:
        return pd.DataFrame()

    group = df.groupby("Week Label", dropna=False).agg(
        Total    = ("Replied", "count"),
        Replied  = ("Replied", "sum"),
        Meeting  = ("Meeting", "sum"),
    ).reset_index()

    group["Reply rate %"]   = (group["Replied"] / group["Total"] * 100).round(1)
    group["Meeting rate %"] = (group["Meeting"] / group["Replied"].replace(0, np.nan) * 100).round(1)

    return group


def temporal_analysis(df: pd.DataFrame) -> dict:
    """
    Analyzes outreach patterns over Contact Month.
    """
    monthly = monthly_breakdown(df)
    weekly = weekly_analysis(df)

    # Best month for replies
    best_month = monthly.loc[monthly["Reply rate %"].idxmax(), "Contact Month"] \
        if not monthly.empty else "N/A"

    return {
        "monthly_breakdown": monthly,
        "weekly_breakdown": weekly,
        "best_month": best_month,
        "insight": (
            f"Most effective outreach month: **{best_month}** "
            f"based on reply rate." if best_month != "N/A" else "Insufficient monthly data."
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

    # Segment comparison
    if "Lead Type" in df.columns:
        clients = segment_funnel(df, "Client")
        partners = segment_funnel(df, "Partner")
        
        if clients["count"] > 0:
            insights.append(
                f"**Clients** (n={clients['count']}): "
                f"{clients['stats']['reply_rate']}% reply rate"
            )
        if partners["count"] > 0:
            insights.append(
                f"**Partners** (n={partners['count']}): "
                f"{partners['stats']['reply_rate']}% reply rate"
            )

    # Best channel
    ch = channel_breakdown(df)
    if not ch.empty:
        best_ch = ch.iloc[0]
        insights.append(
            f"**{best_ch['Contact Channel']}** is most effective "
            f"({best_ch['Reply rate %']}% reply rate, n={int(best_ch['Total'])})"
        )

    # Best decision level
    dl = decision_level_breakdown(df)
    if not dl.empty:
        best_dl = dl.iloc[0]
        worst_dl = dl.iloc[-1]
        insights.append(
            f"Decision level matters: **{best_dl['Decision Level']}** replies {best_dl['Reply rate %']}% "
            f"vs **{worst_dl['Decision Level']}** {worst_dl['Reply rate %']}%"
        )

    # Best industry
    ind = industry_breakdown(df)
    if not ind.empty:
        best_ind = ind.iloc[0]
        insights.append(
            f"**{best_ind['Industry']}** is most responsive "
            f"({best_ind['Reply rate %']}% reply rate)"
        )

    return insights


def segment_comparison(df: pd.DataFrame) -> dict:
    """
    Side-by-side comparison of Client vs Partner segments.
    """
    if "Lead Type" not in df.columns:
        return {}
    
    comparison = {}
    
    for segment in ["Client", "Partner"]:
        df_seg = df[df["Lead Type"] == segment]
        if len(df_seg) == 0:
            continue
        
        comparison[segment] = {
            "count": len(df_seg),
            "funnel": funnel_stats(df_seg),
            "top_channel": channel_breakdown(df_seg).iloc[0] if not channel_breakdown(df_seg).empty else None,
            "top_decision_level": decision_level_breakdown(df_seg).iloc[0] if not decision_level_breakdown(df_seg).empty else None,
        }
    
    return comparison
