"""
learning_loop.py — Close the post-event learning loop.

The decisions log records what SmartFlow recommended and what actually happened,
but logging alone is an *open* loop — nothing reads it back or adapts. This module
closes it:

  1. evaluate_decisions() — read decisions_log.csv back and measure how well past
     recommendations matched reality (clearance MAE, severity accuracy).
  2. record_snapshot()    — append the current model metrics + that recommendation
     accuracy to metrics_history.csv, so drift is visible across retrains.
  3. retrain_and_record() — re-fit the models from the (growing) feature set and
     record a fresh snapshot. This is the nightly job the brief asks for.

`metrics_history.csv` is the artefact that proves the loop is real: one row per
retrain, charted on the Feedback Loop page.

Honest scope: the re-fit reads the canonical feature set (features.csv), which
grows as the pipeline ingests newly *resolved* events — that is how logged
outcomes re-enter training, since a raw decisions_log row lacks the engineered
features a model needs. The decisions log itself supplies the recommendation-vs-
actual accuracy signal that tells you *when* drift warrants a retrain.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from outcomes_log import load_decisions, summary

logger = logging.getLogger(__name__)

_HISTORY_FIELDS = [
    "recorded_at", "n_decisions", "n_outcomes",
    "rec_clearance_mae", "rec_severity_accuracy",
    "closure_roc_auc", "closure_pr_auc",
    "severity_test_acc", "severity_baseline_acc",
    "duration_mae", "duration_baseline_mae",
]


def _project_root(project_root: Path | None) -> Path:
    return project_root or Path(__file__).resolve().parents[1]


def history_path(project_root: Path | None = None) -> Path:
    return _project_root(project_root) / "data" / "processed" / "metrics_history.csv"


def evaluate_decisions(project_root: Path | None = None) -> dict:
    """
    Read the decisions log back and measure recommendation-vs-actual accuracy.

    Extends outcomes_log.summary() (clearance MAE) with severity accuracy — the
    'did our triage call match reality?' signal.
    """
    s = summary(project_root)
    out = {
        "n_decisions": s["total"],
        "n_outcomes": s["with_outcome"],
        "clearance_mae": s["mae"],
        "severity_accuracy": None,
    }
    dec = load_decisions(project_root)
    if not dec.empty:
        sev = dec.dropna(subset=["predicted_severity", "actual_severity"]).copy()
        sev = sev[(sev["actual_severity"].astype(str).str.strip() != "")]
        if len(sev):
            match = (sev["predicted_severity"].astype(str).str.strip()
                     == sev["actual_severity"].astype(str).str.strip())
            out["severity_accuracy"] = round(float(match.mean()) * 100, 1)
    return out


def _model_metrics(project_root: Path | None = None) -> dict:
    """Pull the stored holdout metrics from the trained model artefacts."""
    models = _project_root(project_root) / "models"
    m: dict = {}

    def _load(name):
        p = models / name
        return joblib.load(p) if p.exists() else {}

    clo = _load("closure_predictor.pkl")
    clf = _load("severity_classifier.pkl")
    dur = _load("duration_predictor.pkl")
    m["closure_roc_auc"] = clo.get("roc_auc")
    m["closure_pr_auc"] = clo.get("pr_auc")
    m["severity_test_acc"] = clf.get("test_accuracy")
    m["severity_baseline_acc"] = clf.get("baseline_accuracy")
    m["duration_mae"] = dur.get("mae")
    m["duration_baseline_mae"] = dur.get("baseline_mae")
    return m


def record_snapshot(project_root: Path | None = None,
                    stamp: str | None = None) -> dict:
    """Append a metrics + recommendation-accuracy snapshot to metrics_history.csv."""
    ev = evaluate_decisions(project_root)
    mm = _model_metrics(project_root)
    row = {
        "recorded_at": stamp or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "n_decisions": ev["n_decisions"],
        "n_outcomes": ev["n_outcomes"],
        "rec_clearance_mae": ev["clearance_mae"],
        "rec_severity_accuracy": ev["severity_accuracy"],
        **mm,
    }
    path = history_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=_HISTORY_FIELDS)
    out = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    out.reindex(columns=_HISTORY_FIELDS).to_csv(path, index=False)
    logger.info("Recorded metrics snapshot → %s (now %d rows)", path.name, len(out))
    return row


def load_history(project_root: Path | None = None) -> pd.DataFrame:
    path = history_path(project_root)
    if not path.exists():
        return pd.DataFrame(columns=_HISTORY_FIELDS)
    return pd.read_csv(path)


def retrain_and_record(project_root: Path | None = None) -> dict:
    """Re-fit all models, then record a fresh metrics snapshot. The nightly job."""
    from model_training import run_training
    logger.info("Retraining models from the current feature set …")
    run_training(project_root)
    return record_snapshot(project_root)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    ap = argparse.ArgumentParser(description="Close the SmartFlow learning loop.")
    ap.add_argument("--no-retrain", action="store_true",
                    help="Only record a metrics snapshot; skip the model re-fit.")
    args = ap.parse_args()

    snap = record_snapshot() if args.no_retrain else retrain_and_record()
    print("Snapshot recorded:")
    for k, v in snap.items():
        print(f"  {k}: {v}")
