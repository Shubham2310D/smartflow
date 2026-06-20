"""
test_impact_score.py — Guard the transparent disruption-impact heuristic.

Asserts the score is bounded 0–100, monotonic in its drivers, and flagged as a
heuristic (not a learned forecast).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from impact_score import impact_score


def test_bounded_and_flagged_heuristic():
    lo = impact_score(road_closure=False, corridor_7d=0, is_peak=False)
    hi = impact_score(road_closure=True, corridor_7d=100, is_peak=True)
    assert 0 <= lo["score"] <= 100 and 0 <= hi["score"] <= 100
    assert lo["is_heuristic"] is True
    assert lo["label"] == "Low" and hi["label"] == "Severe"


def test_closure_dominates_and_is_monotonic():
    base = impact_score(road_closure=False, corridor_7d=10, is_peak=False)
    with_closure = impact_score(road_closure=True, corridor_7d=10, is_peak=False)
    assert with_closure["score"] > base["score"]


def test_closure_probability_used_when_not_flagged():
    low_p = impact_score(road_closure=False, corridor_7d=5, is_peak=False, closure_prob=0.1)
    high_p = impact_score(road_closure=False, corridor_7d=5, is_peak=False, closure_prob=0.9)
    assert high_p["score"] > low_p["score"]
