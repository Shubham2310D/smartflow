"""
outcomes_log.py — Persistent decision + outcome log (the post-event learning loop).

Every recommendation SmartFlow produces can be logged here, and an operator can
later record what actually happened (real clearance time, real severity, whether
the plan was followed).  This is what closes the loop the problem statement asks
for: a predicted-vs-actual record that accumulates over time and becomes the
basis for periodic retraining.

The store is a simple append-only CSV — no database needed for a demo, and it is
trivial to inspect, back up, or feed back into model_training.py.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Column order for the log file
FIELDS = [
    "logged_at",                 # ISO timestamp the decision was recorded
    "event_cause",
    "zone",
    "hour_of_day",
    "predicted_severity",
    "predicted_clearance_min",
    "recommended_personnel",
    "actual_clearance_min",      # filled in later by the operator (may be blank)
    "actual_severity",           # filled in later by the operator (may be blank)
    "followed",                  # was the recommendation followed? (yes/no/blank)
]


def get_log_path(project_root: Path | None = None) -> Path:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]
    return project_root / "data" / "processed" / "decisions_log.csv"


def load_decisions(project_root: Path | None = None) -> pd.DataFrame:
    """Return the decisions log as a DataFrame (empty with correct columns if none)."""
    path = get_log_path(project_root)
    if not path.exists():
        return pd.DataFrame(columns=FIELDS)
    df = pd.read_csv(path)
    # Ensure all expected columns exist even if an older file is shorter
    for col in FIELDS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[FIELDS]


def log_decision(record: dict, project_root: Path | None = None) -> None:
    """
    Append one decision to the log.

    `record` may contain any subset of FIELDS; missing keys are stored blank.
    `logged_at` is stamped automatically if not supplied.
    """
    path = get_log_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)

    row = {k: record.get(k, "") for k in FIELDS}
    if not row.get("logged_at"):
        row["logged_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    df_existing = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=FIELDS)
    df_new = pd.concat([df_existing, pd.DataFrame([row])], ignore_index=True)
    df_new[FIELDS].to_csv(path, index=False)


def summary(project_root: Path | None = None) -> dict:
    """Aggregate stats over decisions that have a recorded actual clearance."""
    df = load_decisions(project_root)
    out = {"total": len(df), "with_outcome": 0, "mae": None, "within_30pct": None}
    if df.empty:
        return out

    closed = df.dropna(subset=["actual_clearance_min", "predicted_clearance_min"])
    closed = closed[
        pd.to_numeric(closed["actual_clearance_min"], errors="coerce").notna()
    ].copy()
    if closed.empty:
        return out

    actual = pd.to_numeric(closed["actual_clearance_min"], errors="coerce")
    pred   = pd.to_numeric(closed["predicted_clearance_min"], errors="coerce")
    err    = (actual - pred).abs()
    out["with_outcome"]  = int(len(closed))
    out["mae"]           = float(err.mean())
    out["within_30pct"]  = float(((err / actual.clip(lower=1)) <= 0.30).mean() * 100)
    return out
