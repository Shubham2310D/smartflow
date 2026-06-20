"""
test_learning_loop.py — Guard the post-event learning loop.

Asserts the loop reads decisions back (recommendation-vs-actual accuracy) and that
record_snapshot appends a well-formed row to metrics_history.csv. Uses a temp
project root so it never touches the real logs/artefacts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

import learning_loop as ll
from outcomes_log import log_decision


def _seed(root: Path):
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(parents=True, exist_ok=True)
    # Two logged decisions with recorded actuals: one severity match, one miss.
    log_decision({"event_cause": "accident", "zone": "Z", "hour_of_day": 9,
                  "predicted_severity": "High", "predicted_clearance_min": 60,
                  "actual_clearance_min": 50, "actual_severity": "High"}, project_root=root)
    log_decision({"event_cause": "protest", "zone": "Z", "hour_of_day": 18,
                  "predicted_severity": "Medium", "predicted_clearance_min": 40,
                  "actual_clearance_min": 90, "actual_severity": "High"}, project_root=root)


def test_evaluate_decisions_measures_accuracy(tmp_path):
    _seed(tmp_path)
    ev = ll.evaluate_decisions(project_root=tmp_path)
    assert ev["n_outcomes"] == 2
    assert ev["severity_accuracy"] == 50.0          # 1 of 2 matched
    assert ev["clearance_mae"] == 30.0              # (|60-50| + |40-90|)/2


def test_record_snapshot_appends_history(tmp_path):
    _seed(tmp_path)
    ll.record_snapshot(project_root=tmp_path, stamp="2026-01-01 00:00:00")
    ll.record_snapshot(project_root=tmp_path, stamp="2026-01-02 00:00:00")
    hist = ll.load_history(project_root=tmp_path)
    assert len(hist) == 2
    assert set(["recorded_at", "rec_severity_accuracy", "n_outcomes"]).issubset(hist.columns)
    assert hist.iloc[-1]["rec_severity_accuracy"] == 50.0
