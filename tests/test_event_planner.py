"""
test_event_planner.py — Guard the advance event planner & analog backoff.

Asserts the planner returns a coherent plan (severity + deployment + confidence),
that confidence degrades with fewer analogs, and that construction is treated as
an obstruction rather than a crowd event.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from event_analog import PLANNABLE_EVENTS, _confidence
from event_planner import plan_event


def test_confidence_degrades_with_evidence():
    assert _confidence(0, local=True) == "none"
    assert _confidence(12, local=True) == "high"
    assert _confidence(12, local=False) == "medium"   # citywide caps at medium
    assert _confidence(2, local=True) == "low"


def test_plan_has_required_shape_for_all_plannable_types():
    for cause in PLANNABLE_EVENTS:
        p = plan_event(cause, zone="Central Zone 1", hour=18)
        assert p["expected_severity"] in {"High", "Medium", "Low"}
        assert p["confidence"] in {"high", "medium", "low", "none"}
        assert p["recommendation"]["personnel_count"] >= 1
        assert "barricade_required" in p["recommendation"]


def test_construction_flagged_as_obstruction():
    assert plan_event("construction", zone="Central Zone 1")["is_obstruction"] is True
    assert plan_event("procession", zone="Central Zone 1")["is_obstruction"] is False
