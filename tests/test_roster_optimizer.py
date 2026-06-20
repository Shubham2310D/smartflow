"""
test_roster_optimizer.py — Guard the multi-event roster allocation.

Asserts the min-cost-flow allocator (1) conserves demand, (2) covers everything
when the roster is sufficient, and (3) sacrifices the lowest-priority demand
first when the roster is scarce. Uses synthetic events so it's deterministic and
doesn't depend on the processed CSVs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roster_optimizer import optimize

_LOCS = {"S": (12.97, 77.59)}  # single station co-located with the events


def _events():
    return [
        {"id": "hi", "lat": 12.97, "lon": 77.59, "severity": "High",   "demand": 3},
        {"id": "lo", "lat": 12.97, "lon": 77.59, "severity": "Low",    "demand": 3},
    ]


def test_demand_is_conserved():
    r = optimize(_events(), {"S": 4}, _LOCS)
    assert r["met"] + sum(r["unmet"].values()) == r["total_demand"] == 6
    assert r["officers_used"] == r["met"]


def test_sufficient_roster_covers_all():
    r = optimize(_events(), {"S": 10}, _LOCS)
    assert sum(r["unmet"].values()) == 0
    assert r["met"] == 6


def test_scarce_roster_serves_high_priority_first():
    # Only 3 officers for 6 units of demand → High fully served, Low fully unmet.
    r = optimize(_events(), {"S": 3}, _LOCS)
    assert r["unmet"]["hi"] == 0
    assert r["unmet"]["lo"] == 3


def test_no_stations_means_everything_unmet():
    r = optimize(_events(), {}, _LOCS)
    assert r["met"] == 0
    assert sum(r["unmet"].values()) == 6
