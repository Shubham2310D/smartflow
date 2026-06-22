"""
test_online_cluster.py — Guard real-time cluster assignment + persistence.

A live incident added through the API must get a cluster_label without re-running
DBSCAN, so it joins the right hotspot footprint (and convex hull) on Live Ops.
These assert: centroids load, a point inside a cluster's footprint snaps to it, a
far/empty point is noise, and the event store round-trips the cluster value.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hotspot_engine as he
import event_store as es

_ROOT = Path(__file__).resolve().parents[1]


def test_centroids_have_expected_shape():
    he._CENTROIDS_CACHE = None
    cents = he.cluster_centroids(_ROOT)
    assert isinstance(cents, list) and len(cents) > 0
    assert {"cluster", "lat", "lon", "radius_km"} <= set(cents[0])


def test_point_at_centroid_snaps_to_its_cluster():
    he._CENTROIDS_CACHE = None
    c = he.cluster_centroids(_ROOT)[0]
    assert he.assign_cluster(c["lat"], c["lon"], project_root=_ROOT) == c["cluster"]


def test_far_or_missing_point_is_noise():
    assert he.assign_cluster(0.0, 0.0, project_root=_ROOT) == -1      # off the map
    assert he.assign_cluster(None, None, project_root=_ROOT) == -1     # no location


def test_event_store_round_trips_cluster(tmp_path):
    (tmp_path / "data" / "processed").mkdir(parents=True)
    es.record_event({"id": "x1", "ts": "2024-03-01 10:00:00", "corridor": "ORR",
                     "latitude": 12.97, "longitude": 77.59, "severity": "High",
                     "status": "active", "cluster": 7}, project_root=tmp_path)
    act = es.active_events(project_root=tmp_path)
    assert "cluster" in act.columns and int(act.iloc[0]["cluster"]) == 7
