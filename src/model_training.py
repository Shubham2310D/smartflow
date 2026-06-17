"""
model_training.py — Train XGBoost severity classifier + duration predictor.

Outputs (saved to models/):
  severity_classifier.pkl  — XGBClassifier + metadata
  duration_predictor.pkl   — XGBRegressor + metadata
  shap_severity_summary.png
  shap_duration_summary.png
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Severity classifier — contextual signals only.
# cause/closure are excluded because severity_class is derived from them;
# including them makes the model circular (99% accuracy, zero predictive value).
CLF_FEATURES = [
    "hour_of_day",
    "day_of_week",
    "month",
    "is_peak_hour",
    "is_weekend",
    "junction_repeat_count",
    "corridor_7d_score",
    "veh_type_encoded",
]

# Duration predictor — cause/closure are legitimate here.
# duration_minutes is an observed real value, not derived from these fields.
# Road closures and accident causes genuinely take longer to clear.
REG_FEATURES = [
    "cause_severity_weight",
    "road_closure_binary",
    "hour_of_day",
    "day_of_week",
    "month",
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
    df = pd.read_csv(path)
    df["severity_label"] = df["severity_class"].map(SEVERITY_LABEL_MAP)
    all_features = list(dict.fromkeys(CLF_FEATURES + REG_FEATURES))
    df = df.dropna(subset=all_features + ["severity_class"]).reset_index(drop=True)
    logger.info("Training dataset: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Severity classifier
# ---------------------------------------------------------------------------

def train_severity_classifier(df: pd.DataFrame, models_dir: Path) -> XGBClassifier:
    X = df[CLF_FEATURES].fillna(0).astype(float)
    y = df["severity_label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    sample_weights = compute_sample_weight("balanced", y_train)
    clf = XGBClassifier(**XGB_CLF_PARAMS)
    clf.fit(X_train, y_train, sample_weight=sample_weights)

    y_pred = clf.predict(X_test)
    report = classification_report(
        y_test, y_pred, target_names=["Low", "Medium", "High"]
    )
    logger.info("Severity classifier report:\n%s", report)
    logger.info("Confusion matrix:\n%s", confusion_matrix(y_test, y_pred))

    # 5-fold stratified CV accuracy
    cv_scores = cross_val_score(clf, X, y, cv=StratifiedKFold(5), scoring="accuracy")
    logger.info("5-fold CV accuracy: %.3f ± %.3f", cv_scores.mean(), cv_scores.std())

    payload = {
        "model":           clf,
        "features":        CLF_FEATURES,
        "label_map":       SEVERITY_LABEL_MAP,
        "inverse_map":     SEVERITY_INVERSE_MAP,
        "colors":          SEVERITY_COLORS,
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std":  float(cv_scores.std()),
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

    X = df_reg[REG_FEATURES].fillna(0).astype(float)
    # Log1p transform to handle right-skewed distribution
    y_raw = df_reg["duration_minutes"].astype(float)
    y     = np.log1p(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    reg = XGBRegressor(**XGB_REG_PARAMS)
    reg.fit(X_train, y_train)

    y_pred_log = reg.predict(X_test)
    y_pred     = np.expm1(y_pred_log)
    y_test_raw = np.expm1(y_test.values)
    mae   = mean_absolute_error(y_test_raw, y_pred)
    rmse  = float(np.sqrt(np.mean((y_test_raw - y_pred) ** 2)))
    r2    = r2_score(y_test_raw, y_pred)
    logger.info(
        "Duration predictor — MAE: %.1f min, RMSE: %.1f min, R²: %.3f",
        mae, rmse, r2
    )

    payload = {
        "model":        reg,
        "features":     REG_FEATURES,
        "log_transform": True,   # predictions must be expm1'd
        "mae":          mae,
        "rmse":         rmse,
        "r2":           r2,
    }
    joblib.dump(payload, models_dir / "duration_predictor.pkl")
    logger.info("Saved duration_predictor.pkl")

    _save_shap_summary(
        reg, X_test, REG_FEATURES, models_dir / "shap_duration_summary.png",
        is_multiclass=False
    )

    return reg


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
    train_duration_predictor(df, models_dir)
    logger.info("Training complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    run_training()
