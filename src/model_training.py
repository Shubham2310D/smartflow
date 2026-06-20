"""
model_training.py — Train XGBoost severity classifier + duration predictor.

Outputs (saved to models/):
  severity_classifier.pkl  — XGBClassifier + metadata
  duration_predictor.pkl   — XGBRegressor + metadata
  shap_severity_summary.png
  shap_duration_summary.png
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reproducibility — record the library versions a model was trained with, so a
# version-skewed environment (which can silently break or mis-load a pickle) is
# detected rather than failing mysteriously. Stamped into every model payload;
# checked at load time (API /health and the dashboard).
# ---------------------------------------------------------------------------

# Libraries whose version actually governs pickle/predict compatibility.
_COMPAT_LIBS = ("scikit-learn", "xgboost", "numpy")


def lib_versions() -> dict:
    """Versions of the libraries that matter for model (de)serialisation."""
    return {
        "python": sys.version.split()[0],
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    }


def check_lib_versions(payload: dict) -> list[str]:
    """
    Return human-readable warnings for any compatibility-critical library whose
    runtime version differs from what the model was trained with. Empty list
    means the environment matches (or the model predates version stamping).
    """
    trained = (payload or {}).get("lib_versions") or {}
    if not trained:
        return ["model has no recorded training versions — retrain to embed them"]
    current = lib_versions()
    return [
        f"{lib}: trained on {trained[lib]}, running {current[lib]}"
        for lib in _COMPAT_LIBS
        if trained.get(lib) and current.get(lib) and trained[lib] != current[lib]
    ]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Severity classifier — contextual signals only.
# cause/closure are excluded because severity_class is derived from them;
# including them makes the model circular (99% accuracy, zero predictive value).
CLF_FEATURES = [
    "hour_of_day",
    "day_of_week",
    # "month" dropped: coverage is only Nov 2023–Apr 2024 (~5 months), so under a
    # chronological split the test months barely appear in train — month can't
    # learn seasonality from this window and is mildly leaky. Confirmed
    # AUC/accuracy-neutral on removal.
    "is_peak_hour",
    "is_weekend",
    "junction_repeat_count",
    "corridor_7d_score",
    "veh_type_encoded",
]

# Duration predictor — cause/closure are legitimate here.
# duration_minutes is an observed real value, not derived from these fields.
# Road closures and accident causes genuinely take longer to clear.
# event_semantic_encoded (derived from the free-text description) is included
# ONLY here, never in the severity classifier: severity's label is partly
# derived from cause, so a text-derived cause-proxy would re-introduce leakage
# on the classifier — but for an observed target like duration it is legitimate.
REG_FEATURES = [
    "cause_severity_weight",
    "road_closure_binary",
    "event_semantic_encoded",
    "hour_of_day",
    "day_of_week",
    # "month" dropped — see CLF_FEATURES (5-month window, leaky under time split).
    "is_peak_hour",
    "is_weekend",
    "junction_repeat_count",
    "corridor_7d_score",
    "veh_type_encoded",
]

# Road-closure predictor — the genuinely learnable, operationally meaningful,
# pre-event target.  requires_road_closure is OBSERVED (not a derived label) and
# is exactly what drives barricading/diversion planning.  cause IS allowed here
# because closure is not defined from cause (no circularity).  The target is the
# 7.4%-positive minority class, so we report ROC-AUC / PR-AUC / recall, never bare
# accuracy (which a "always predict no-closure" baseline already scores ~93% on).
CLOSURE_FEATURES = [
    "cause_severity_weight",
    "event_semantic_encoded",
    "hour_of_day",
    "day_of_week",
    # "month" dropped — see CLF_FEATURES (5-month window, leaky under time split).
    "is_peak_hour",
    "is_weekend",
    "junction_repeat_count",
    "corridor_7d_score",
    "veh_type_encoded",
]

# Backward-compat alias used by dashboard imports
FEATURES = CLF_FEATURES

SEVERITY_LABEL_MAP    = {"Low": 0, "Medium": 1, "High": 2}
SEVERITY_INVERSE_MAP  = {0: "Low", 1: "Medium", 2: "High"}
SEVERITY_COLORS       = {"Low": "#28a745", "Medium": "#ffc107", "High": "#dc3545"}

XGB_CLF_PARAMS = dict(
    n_estimators     = 300,
    max_depth        = 5,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    objective        = "multi:softprob",   # num_class inferred automatically in XGBoost 3.x
    eval_metric      = "mlogloss",
    n_jobs           = -1,
    random_state     = 42,
)

XGB_REG_PARAMS = dict(
    n_estimators     = 300,
    max_depth        = 5,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    objective        = "reg:squarederror",
    n_jobs           = -1,
    random_state     = 42,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_model_ready(project_root: Path) -> pd.DataFrame:
    path = project_root / "data" / "processed" / "model_ready.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run feature_engineering.py first."
        )
    df = pd.read_csv(path, parse_dates=["start_datetime"])
    df["severity_label"] = df["severity_class"].map(SEVERITY_LABEL_MAP)
    all_features = list(dict.fromkeys(CLF_FEATURES + REG_FEATURES))
    df = df.dropna(subset=all_features + ["severity_class"]).reset_index(drop=True)
    # Sort chronologically so train/test splits respect time order (no leakage).
    if "start_datetime" in df.columns:
        df = df.sort_values("start_datetime", kind="stable").reset_index(drop=True)
    logger.info("Training dataset: %d rows", len(df))
    return df


def _chronological_split(X: pd.DataFrame, y, test_frac: float = 0.2):
    """
    Split into train (earlier events) and test (later events) by row order.
    Assumes the input is already sorted chronologically.  This is the honest
    way to validate an operational forecasting model: you only ever train on
    the past and predict the future, never the reverse.
    """
    n = len(X)
    cut = int(n * (1 - test_frac))
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


# ---------------------------------------------------------------------------
# Severity classifier
# ---------------------------------------------------------------------------

def train_severity_classifier(df: pd.DataFrame, models_dir: Path) -> XGBClassifier:
    X = df[CLF_FEATURES].fillna(0).astype(float)
    y = df["severity_label"].astype(int)

    # Honest, leakage-free validation: train on earlier events, test on later.
    X_train, X_test, y_train, y_test = _chronological_split(X, y, test_frac=0.2)

    sample_weights = compute_sample_weight("balanced", y_train)
    clf = XGBClassifier(**XGB_CLF_PARAMS)
    clf.fit(X_train, y_train, sample_weight=sample_weights)

    y_pred = clf.predict(X_test)
    from sklearn.metrics import accuracy_score, f1_score
    test_acc = float(accuracy_score(y_test, y_pred))
    test_f1  = float(f1_score(y_test, y_pred, average="macro"))
    # Majority-class baseline on the test window, for honest comparison
    baseline = float((y_test == y_train.mode().iloc[0]).mean())
    report = classification_report(
        y_test, y_pred, target_names=["Low", "Medium", "High"], zero_division=0
    )
    logger.info("Severity classifier (chronological holdout) report:\n%s", report)
    logger.info("Confusion matrix:\n%s", confusion_matrix(y_test, y_pred))
    logger.info(
        "Chronological holdout — accuracy: %.3f, macro-F1: %.3f, "
        "majority baseline: %.3f", test_acc, test_f1, baseline,
    )

    # Random 5-fold CV — OPTIMISTIC (ignores time order); reported for reference only.
    cv_scores = cross_val_score(clf, X, y, cv=StratifiedKFold(5), scoring="accuracy")
    logger.info(
        "Random 5-fold CV accuracy (optimistic): %.3f ± %.3f",
        cv_scores.mean(), cv_scores.std(),
    )

    payload = {
        "model":           clf,
        "features":        CLF_FEATURES,
        "label_map":       SEVERITY_LABEL_MAP,
        "inverse_map":     SEVERITY_INVERSE_MAP,
        "colors":          SEVERITY_COLORS,
        "test_accuracy":   test_acc,            # headline: chronological holdout
        "test_macro_f1":   test_f1,
        "baseline_accuracy": baseline,
        "cv_accuracy_mean": float(cv_scores.mean()),   # optimistic, reference only
        "cv_accuracy_std":  float(cv_scores.std()),
        "lib_versions":     lib_versions(),
    }
    joblib.dump(payload, models_dir / "severity_classifier.pkl")
    logger.info("Saved severity_classifier.pkl")

    _save_shap_summary(
        clf, X_test, CLF_FEATURES, models_dir / "shap_severity_summary.png",
        is_multiclass=True, class_names=["Low", "Medium", "High"]
    )

    return clf


# ---------------------------------------------------------------------------
# Duration predictor
# ---------------------------------------------------------------------------

def train_duration_predictor(df: pd.DataFrame, models_dir: Path) -> XGBRegressor:
    df_reg = df.dropna(subset=["duration_minutes"]).copy()
    # Cap extreme durations at 24 h before log-transform to remove stale-ticket noise
    df_reg = df_reg[df_reg["duration_minutes"] <= 1440].copy()
    logger.info("Regression training on %d rows (capped ≤ 1440 min)", len(df_reg))

    # Keep chronological order so the split trains on the past, tests on the future.
    df_reg = df_reg.sort_values("start_datetime", kind="stable").reset_index(drop=True) \
        if "start_datetime" in df_reg.columns else df_reg
    X = df_reg[REG_FEATURES].fillna(0).astype(float)
    # Log1p transform to handle right-skewed distribution
    y_raw = df_reg["duration_minutes"].astype(float)
    y     = np.log1p(y_raw)

    X_train, X_test, y_train, y_test = _chronological_split(X, y, test_frac=0.2)

    reg = XGBRegressor(**XGB_REG_PARAMS)
    reg.fit(X_train, y_train)

    y_pred_log = reg.predict(X_test)
    y_pred     = np.expm1(y_pred_log)
    y_test_raw = np.expm1(y_test.values)
    mae   = mean_absolute_error(y_test_raw, y_pred)
    rmse  = float(np.sqrt(np.mean((y_test_raw - y_pred) ** 2)))
    r2    = r2_score(y_test_raw, y_pred)

    # Naive baseline: always predict the training median (the MAE-optimal
    # constant for a skewed target).  The model only "earns its place" if it
    # beats this — report both so the MAE is never read in isolation.
    train_median  = float(np.expm1(y_train).median())
    baseline_mae  = mean_absolute_error(y_test_raw, np.full_like(y_test_raw, train_median))
    lift = baseline_mae - mae
    logger.info(
        "Duration predictor — MAE: %.1f min, RMSE: %.1f min, R²: %.3f",
        mae, rmse, r2,
    )
    logger.info(
        "Duration baseline (predict-median=%.0f) MAE: %.1f min  →  model lift: %+.1f min",
        train_median, baseline_mae, lift,
    )

    payload = {
        "model":        reg,
        "features":     REG_FEATURES,
        "log_transform": True,   # predictions must be expm1'd
        "mae":          mae,
        "rmse":         rmse,
        "r2":           r2,
        "baseline_mae": float(baseline_mae),
        "baseline_median": train_median,
        "lift_vs_baseline": float(lift),
        "lib_versions": lib_versions(),
    }
    joblib.dump(payload, models_dir / "duration_predictor.pkl")
    logger.info("Saved duration_predictor.pkl")

    # Save the honest out-of-sample (chronological holdout) predicted-vs-actual
    # pairs so the Feedback Loop dashboard can show real calibration, not an
    # in-sample fit.
    backtest = pd.DataFrame({
        "actual_minutes":    np.round(y_test_raw, 1),
        "predicted_minutes": np.round(y_pred, 1),
    })
    backtest_path = models_dir.parent / "data" / "processed" / "duration_backtest.csv"
    backtest_path.parent.mkdir(parents=True, exist_ok=True)
    backtest.to_csv(backtest_path, index=False)
    logger.info("Saved duration_backtest.csv (%d holdout rows)", len(backtest))

    _save_shap_summary(
        reg, X_test, REG_FEATURES, models_dir / "shap_duration_summary.png",
        is_multiclass=False
    )

    return reg


# ---------------------------------------------------------------------------
# Road-closure predictor (real, observed, pre-event target)
# ---------------------------------------------------------------------------

def train_closure_predictor(df: pd.DataFrame, models_dir: Path):
    """
    Predict P(requires_road_closure) — a real observed outcome, not a synthetic
    label.  Chronological split, class-weighted XGBoost, isotonic-calibrated so
    the probability is trustworthy.  Reported against the base rate, with
    ROC-AUC / PR-AUC / recall (accuracy is meaningless at 7.4% positives).
    """
    X = df[CLOSURE_FEATURES].fillna(0).astype(float)
    y = df["road_closure_binary"].astype(int)

    X_train, X_test, y_train, y_test = _chronological_split(X, y, test_frac=0.2)
    base_rate = float(y_test.mean())

    # Handle imbalance via scale_pos_weight on the training window
    pos = max(int(y_train.sum()), 1)
    neg = int((y_train == 0).sum())
    spw = neg / pos

    params = dict(XGB_CLF_PARAMS)
    params.pop("objective", None)
    params.pop("eval_metric", None)
    base = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        scale_pos_weight=spw, **params,
    )

    # Calibrate on the most recent slice of the training window (prefit), so the
    # probabilities the dashboard shows are honest, not raw XGBoost scores.
    cut = int(len(X_train) * 0.8)
    base.fit(X_train.iloc[:cut], y_train.iloc[:cut])
    clf = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
    clf.fit(X_train.iloc[cut:], y_train.iloc[cut:])

    proba = clf.predict_proba(X_test)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    roc   = float(roc_auc_score(y_test, proba)) if y_test.nunique() > 1 else float("nan")
    pr    = float(average_precision_score(y_test, proba)) if y_test.nunique() > 1 else float("nan")
    tp = int(((pred == 1) & (y_test == 1)).sum())
    fn = int(((pred == 0) & (y_test == 1)).sum())
    fp = int(((pred == 1) & (y_test == 0)).sum())
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0

    logger.info(
        "Closure predictor (chronological holdout) — base rate: %.3f, "
        "ROC-AUC: %.3f, PR-AUC: %.3f, recall: %.2f, precision: %.2f",
        base_rate, roc, pr, recall, precision,
    )

    payload = {
        "model":      clf,
        "features":   CLOSURE_FEATURES,
        "base_rate":  base_rate,
        "roc_auc":    roc,
        "pr_auc":     pr,
        "recall":     recall,
        "precision":  precision,
        "threshold":  0.5,
        "lib_versions": lib_versions(),
    }
    joblib.dump(payload, models_dir / "closure_predictor.pkl")
    logger.info("Saved closure_predictor.pkl")
    return clf


# ---------------------------------------------------------------------------
# Cause-based clearance lookup (honest replacement for the duration "forecast")
# ---------------------------------------------------------------------------

def save_clearance_stats(df: pd.DataFrame, project_root: Path):
    """
    Empirical clearance distribution per cause: median + IQR (p25–p75).

    The duration field is mostly administrative ticket-close time (only ~69 of
    ~2,760 rows come from a real resolved timestamp), and an ML regressor does
    not beat the median on it.  So instead of presenting a false point forecast,
    we surface the honest historical distribution as a typical range.
    """
    import json

    dur = df.dropna(subset=["duration_minutes"]).copy()
    dur = dur[(dur["duration_minutes"] > 0) & (dur["duration_minutes"] <= 1440)]

    stats = {"_overall": _quantile_block(dur["duration_minutes"])}
    for cause, grp in dur.groupby("event_cause"):
        if len(grp) >= 5:
            stats[cause] = _quantile_block(grp["duration_minutes"])

    out = project_root / "data" / "processed" / "clearance_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2))
    logger.info("Saved clearance_stats.json (%d causes + overall)", len(stats) - 1)
    return stats


def _quantile_block(s: pd.Series) -> dict:
    return {
        "median": round(float(s.median()), 1),
        "p25":    round(float(s.quantile(0.25)), 1),
        "p75":    round(float(s.quantile(0.75)), 1),
        "n":      int(len(s)),
    }


# ---------------------------------------------------------------------------
# SHAP summary plot helper
# ---------------------------------------------------------------------------

def _save_shap_summary(
    model, X_test, feature_names: list[str], out_path: Path,
    is_multiclass: bool = False, class_names: list[str] | None = None
):
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test)

        fig, ax = plt.subplots(figsize=(10, 6))
        if is_multiclass and isinstance(shap_vals, list):
            # Use the "High" class SHAP values for the summary
            shap.summary_plot(
                shap_vals[2], X_test,
                feature_names=feature_names,
                show=False,
            )
        else:
            shap.summary_plot(
                shap_vals, X_test,
                feature_names=feature_names,
                show=False,
            )
        plt.title(out_path.stem.replace("_", " ").title())
        plt.tight_layout()
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()
        logger.info("Saved SHAP plot → %s", out_path.name)
    except Exception as exc:
        logger.warning("SHAP plot skipped (%s)", exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_training(project_root: Path | None = None):
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]

    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Run feature engineering if needed
    model_ready = project_root / "data" / "processed" / "model_ready.csv"
    if not model_ready.exists():
        import sys
        sys.path.insert(0, str(project_root / "src"))
        from feature_engineering import run_feature_engineering
        run_feature_engineering(project_root=project_root)

    df = load_model_ready(project_root)
    train_severity_classifier(df, models_dir)
    train_closure_predictor(df, models_dir)
    train_duration_predictor(df, models_dir)
    save_clearance_stats(df, project_root)
    logger.info("Training complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    run_training()
