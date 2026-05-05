import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")
 
 
# ── helpers ───────────────────────────────────────────────────────────────────
 
def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Shared preprocessing step: OneHotEncode categoricals, pass numerics through."""
    cat = X.select_dtypes(include=["object"]).columns.tolist()
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False,
                                  handle_unknown="ignore"), cat),
            ("num", "passthrough", num),
        ],
        remainder="passthrough",
    )
 
 
def _tree_clf(random_state: int = 42) -> DecisionTreeClassifier:
    """Standard Decision Tree with thesis-aligned hyperparameters."""
    return DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=4,
        min_samples_split=8,
        class_weight="balanced",   # keeps class weighting even without SMOTE
        random_state=random_state,
    )
 
 
def _evaluate(y_true, y_pred) -> dict:
    """Returns a tidy metrics dict for one set of predictions."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    pos = (tp + fn)  # true positives in ground-truth
    return {
        "accuracy":   round(accuracy_score(y_true, y_pred)                         * 100, 1),
        "precision":  round(precision_score(y_true, y_pred, zero_division=0)       * 100, 1),
        "recall":     round(recall_score(y_true, y_pred, zero_division=0)           * 100, 1),
        "f1":         round(f1_score(y_true, y_pred, zero_division=0)               * 100, 1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "support_pos": int(pos),
    }
 
 
# ── SMOTE cross-validation ────────────────────────────────────────────────────
 
def smote_cv_comparison(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Compares Decision Tree trained WITH vs WITHOUT SMOTE using stratified K-fold CV.
 
    SMOTE is applied INSIDE each fold on the training fold only — this is the
    correct way to avoid data leakage (Japkowicz & Stephen, 2002; Chawla et al., 2002).
 
    Args:
        X            Feature matrix (may contain categorical columns).
        y            Binary target (0/1).
        n_splits     Number of CV folds (default 5).
        random_state Random seed for reproducibility.
 
    Returns:
        dict with keys "without_smote" and "with_smote", each containing
        per-fold and mean metrics, plus a "comparison" summary dict.
    """
    # ── guard rails ──────────────────────────────────────────────────────────
    if len(X) < n_splits + 5:
        return {"error": f"Too few samples for {n_splits}-fold CV "
                         f"(need ≥ {n_splits + 5}, have {len(X)})."
    if y.nunique() < 2:
        return {"error": "Target has only one class — need both replied (1) "
                         "and not replied (0) to compare models."}
 
    min_minority = int(y.sum())
    if min_minority < 2:
        return {"error": f"Only {min_minority} positive example(s) — "
                         "SMOTE needs ≥ 2 minority samples per fold."}
 
    # ── try importing imbalanced-learn ────────────────────────────────────────
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        return {"error": "imbalanced-learn is not installed. "
                         "Run: pip install imbalanced-learn"}
 
    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True,
                             random_state=random_state)
    pre    = _build_preprocessor(X)
 
    fold_results = {"without": [], "with": []}
 
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr_raw, X_te_raw = X.iloc[train_idx], X.iloc[test_idx]
        y_tr,     y_te     = y.iloc[train_idx], y.iloc[test_idx]
 
        # ── skip folds where test set is all one class ────────────────────
        if y_te.nunique() < 2:
            continue
 
        # ── shared preprocessing (fit on train, transform both) ───────────
        X_tr_enc = pre.fit_transform(X_tr_raw)
        X_te_enc = pre.transform(X_te_raw)
 
        # ── MODEL A: no SMOTE ─────────────────────────────────────────────
        clf_a = _tree_clf(random_state)
        clf_a.fit(X_tr_enc, y_tr)
        fold_results["without"].append(_evaluate(y_te, clf_a.predict(X_te_enc)))
 
        # ── MODEL B: SMOTE on training fold only ──────────────────────────
        # k_neighbors must be < minority class count in fold
        minority_in_fold = int(y_tr.sum())
        k = min(5, minority_in_fold - 1) if minority_in_fold > 1 else 1
        if k < 1:
            # can't apply SMOTE — skip with_smote for this fold
            fold_results["with"].append(_evaluate(y_te, clf_a.predict(X_te_enc)))
            continue
 
        smote = SMOTE(k_neighbors=k, random_state=random_state)
        try:
            X_tr_sm, y_tr_sm = smote.fit_resample(X_tr_enc, y_tr)
        except Exception:
            # fallback: use original if SMOTE fails
            X_tr_sm, y_tr_sm = X_tr_enc, y_tr
 
        clf_b = _tree_clf(random_state)
        clf_b.fit(X_tr_sm, y_tr_sm)
        fold_results["with"].append(_evaluate(y_te, clf_b.predict(X_te_enc)))
 
    # ── aggregate folds ───────────────────────────────────────────────────────
    def _agg(folds: list[dict]) -> dict:
        if not folds:
            return {}
        keys = [k for k in folds[0] if k not in ("tp", "fp", "fn", "tn", "support_pos")]
        out = {}
        for k in keys:
            vals = [f[k] for f in folds]
            out[f"{k}_mean"] = round(float(np.mean(vals)), 1)
            out[f"{k}_std"]  = round(float(np.std(vals)),  1)
        # summed confusion matrix across folds
        out["tp"] = sum(f["tp"] for f in folds)
        out["fp"] = sum(f["fp"] for f in folds)
        out["fn"] = sum(f["fn"] for f in folds)
        out["tn"] = sum(f["tn"] for f in folds)
        out["n_folds_used"] = len(folds)
        return out
 
    agg_no   = _agg(fold_results["without"])
    agg_smote = _agg(fold_results["with"])
 
    # ── comparison deltas ─────────────────────────────────────────────────────
    comparison = {}
    if agg_no and agg_smote:
        for metric in ("accuracy", "precision", "recall", "f1"):
            k = f"{metric}_mean"
            delta = round(agg_smote.get(k, 0) - agg_no.get(k, 0), 1)
            comparison[metric] = {
                "without": agg_no.get(k, 0),
                "with":    agg_smote.get(k, 0),
                "delta":   delta,
                "improved": delta > 0,
            }
 
    return {
        "n_samples":       len(X),
        "n_splits":        n_splits,
        "positive_rate":   round(float(y.mean()) * 100, 1),
        "without_smote":   agg_no,
        "with_smote":      agg_smote,
        "comparison":      comparison,
        "fold_detail":     fold_results,   # raw per-fold data for advanced use
    }
 
 
def smote_full_model(
    X: pd.DataFrame,
    y: pd.Series,
    apply_smote: bool = True,
    random_state: int = 42,
    test_size: float = 0.3,
) -> dict:
    """
    Trains a single Decision Tree on a hold-out split, optionally with SMOTE.
 
    SMOTE is applied ONLY on X_train — X_test is never touched (no leakage).
 
    Args:
        X            Feature matrix.
        y            Binary target.
        apply_smote  Whether to apply SMOTE to the training set.
        random_state Seed for reproducibility.
        test_size    Fraction of data held out for evaluation.
 
    Returns:
        dict with metrics, fitted pipeline, and original-class counts.
    """
    from sklearn.model_selection import train_test_split
 
    if len(X) < 10:
        return {"error": "Need at least 10 samples."}
    if y.nunique() < 2:
        return {"error": "Target has only one class."}
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
 
    pre = _build_preprocessor(X)
    X_tr_enc = pre.fit_transform(X_train)
    X_te_enc = pre.transform(X_test)
 
    class_counts_before = {"majority": int((y_train == 0).sum()),
                           "minority": int((y_train == 1).sum())}
 
    if apply_smote:
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            return {"error": "imbalanced-learn not installed. "
                             "Run: pip install imbalanced-learn"}
 
        minority_count = int(y_train.sum())
        if minority_count < 2:
            return {"error": f"Only {minority_count} positive sample(s) in training set — "
                             "SMOTE requires at least 2."}
 
        k = min(5, minority_count - 1)
        smote = SMOTE(k_neighbors=k, random_state=random_state)
        X_tr_enc, y_train = smote.fit_resample(X_tr_enc, y_train)
 
    class_counts_after = {"majority": int((y_train == 0).sum()),
                          "minority": int((y_train == 1).sum())}
 
    clf = _tree_clf(random_state)
    clf.fit(X_tr_enc, y_train)
 
    y_pred = clf.predict(X_te_enc)
    metrics = _evaluate(y_test, y_pred)
 
    # feature importance (using transformed names from preprocessor)
    try:
        feat_names = pre.get_feature_names_out()
    except Exception:
        feat_names = [f"f{i}" for i in range(clf.n_features_in_)]
 
    fi = pd.DataFrame({
        "Feature":      feat_names,
        "Importance":   clf.feature_importances_,
        "Importance %": (clf.feature_importances_ * 100).round(1),
    }).sort_values("Importance", ascending=False)
 
    return {
        "apply_smote":          apply_smote,
        "metrics":              metrics,
        "feature_importance":   fi,
        "class_counts_before":  class_counts_before,
        "class_counts_after":   class_counts_after,
        "n_train":              len(X_train),
        "n_test":               len(X_test),
    }

# Re-export prepare_tree_data from analytics_redesigned for convenience
from modules.analytics_redesigned import prepare_tree_data
