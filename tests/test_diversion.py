"""
test_diversion.py — Guard the real reroute engine.

Asserts a diversion around a known Bengaluru arterial is feasible, returns a
coherent path with non-negative added distance, and that out-of-network /
nonsensical points degrade gracefully (feasible=False, never an exception).
Skips if the committed OSM road cache is absent.
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import diversion as dv


def _require_cache():
    if not (_ROOT / "data" / "processed" / "osm_roads.json.gz").exists():
        pytest.skip("osm_roads.json.gz not present")


def test_reroute_on_arterial_is_feasible_and_coherent():
    _require_cache()
    r = dv.plan_diversion(12.9568, 77.7011, project_root=_ROOT)  # ORR @ Marathahalli
    assert r["feasible"] is True
    assert r["detour_m"] > 0
    assert r["extra_m"] >= 0                      # a detour is never shorter than straight-through
    assert r["detour_m"] >= r["direct_m"]
    assert len(r["detour_path"]) >= 2
    assert all(len(pt) == 2 for pt in r["detour_path"])
    # Snapped to a real road (within the configured threshold).
    assert r["snap_distance_m"] < 150


def test_point_with_no_roads_degrades_gracefully():
    _require_cache()
    # Far outside the Bengaluru network — must report infeasible, not raise.
    r = dv.plan_diversion(0.0, 0.0, project_root=_ROOT)
    assert r["feasible"] is False
    assert "reason" in r


def test_geometry_helpers():
    # ~111 m per 0.001° latitude near the equator-ish scale used here.
    d = dv._haversine_m((12.97, 77.59), (12.971, 77.59))
    assert 100 < d < 120
    # Opposite bearings → ~180° gap.
    assert dv._angle_gap(0, 180) == pytest.approx(180)
    assert dv._angle_gap(10, 350) == pytest.approx(20)
