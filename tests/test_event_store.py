"""
test_event_store.py — Guard the real-time SQLite event store.

Asserts events round-trip, live history reflects what's been recorded (real
state, not a static proxy), and the active-set query filters by status. Uses a
temp project root so the real DB is untouched.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import event_store as es


def _seed(root: Path):
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    for i in range(3):
        es.record_event({"id": f"e{i}", "ts": f"2024-03-0{i+1} 10:00:00",
                         "corridor": "ORR", "junction": "Marathahalli",
                         "severity": "High", "status": "active"}, project_root=root)
    es.record_event({"id": "old", "ts": "2024-03-04 10:00:00", "corridor": "ORR",
                     "junction": "Marathahalli", "severity": "Low",
                     "status": "closed"}, project_root=root)


def test_records_and_counts(tmp_path):
    _seed(tmp_path)
    assert es.count(project_root=tmp_path) == 4


def test_live_history_reflects_recorded_events(tmp_path):
    _seed(tmp_path)
    h = es.live_history("ORR", junction="Marathahalli", project_root=tmp_path)
    # all 4 events on ORR fall within 7 days of the latest; 4 prior at the junction
    assert h["corridor_7d_score"] >= 3
    assert h["junction_repeat_count"] == 4


def test_live_history_empty_store_returns_blank(tmp_path):
    (tmp_path / "data" / "processed").mkdir(parents=True)
    assert es.live_history("ORR", project_root=tmp_path) == {}


def test_active_events_filters_status(tmp_path):
    _seed(tmp_path)
    act = es.active_events(project_root=tmp_path)
    assert len(act) == 3 and (act["status"] == "active").all()
