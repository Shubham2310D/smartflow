"""
diversion.py — Turn "diversion = True/False" into an actual reroute.

The brief asks for diversion plans; the system previously only raised a boolean
flag. With the cached OSM road network (osm_features.py) we can do the real
thing: build a road graph, treat the incident's road as blocked, and compute the
shortest path *around* it — a concrete detour with an added-distance cost.

How it works
------------
1. Take the road geometry within `radius_km` of the incident (a detour is local,
   so we route on a small subgraph — fast, and the answer is the same).
2. Build a networkx graph: nodes = road coordinates (shared OSM points at
   intersections connect ways), edges weighted by metres.
3. Snap the incident to the nearest node (the blockage). Pick the two
   "through" neighbours — the pair whose bearings are most opposite — as the
   entry/exit of the blocked stretch.
4. Remove the blocked node and run Dijkstra entry→exit on what's left. That path
   is the diversion. Report its length vs. the direct (blocked) distance.

Graceful by design: if the network is unavailable, the road is a dead-end, or no
alternate route exists within the mapped area, it returns feasible=False with a
clear reason instead of raising.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)

_EARTH_R_M = 6_371_000.0
_DEFAULT_RADIUS_KM = 2.0


def _haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_R_M * math.asin(math.sqrt(h))


def _bearing(src: tuple[float, float], dst: tuple[float, float]) -> float:
    lat1, lat2 = math.radians(src[0]), math.radians(dst[0])
    dlon = math.radians(dst[1] - src[1])
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _angle_gap(b1: float, b2: float) -> float:
    """Smallest separation between two bearings (0–180)."""
    d = abs(b1 - b2) % 360
    return 360 - d if d > 180 else d


def _build_local_graph(ways: list[dict], lat: float, lon: float, radius_km: float):
    import networkx as nx  # noqa: PLC0415

    g = nx.Graph()
    dlat = radius_km / 110.574
    dlon = radius_km / (111.320 * math.cos(math.radians(lat)) + 1e-9)
    lo_lat, hi_lat = lat - dlat, lat + dlat
    lo_lon, hi_lon = lon - dlon, lon + dlon

    for w in ways:
        pts = w.get("g", [])
        # Keep a way if any of its points falls in the local window.
        if not any(lo_lat <= p[0] <= hi_lat and lo_lon <= p[1] <= hi_lon for p in pts):
            continue
        for p, q in zip(pts, pts[1:]):
            a, b = (p[0], p[1]), (q[0], q[1])
            if a == b:
                continue
            g.add_edge(a, b, m=_haversine_m(a, b))
    return g


def _nearest_node(g, lat: float, lon: float):
    best, best_d = None, float("inf")
    for node in g.nodes:
        d = _haversine_m((lat, lon), node)
        if d < best_d:
            best, best_d = node, d
    return best, best_d


def plan_diversion(lat: float, lon: float, project_root: Path | None = None,
                   radius_km: float = _DEFAULT_RADIUS_KM,
                   block_radius_m: float = 80.0) -> dict:
    """
    Compute a real detour around a blockage at (lat, lon).

    The closure is modelled as a zone of `block_radius_m` around the incident
    (a single OSM shape-point would give a meaningless ~3 m "block"). We find the
    road's two most-opposite exits from that zone and route between them with the
    zone removed. Returns feasibility, the detour path (list of [lat, lon]), the
    direct-vs-detour distance in metres, and the added cost — or feasible=False
    with a reason.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]

    # Reuse the same cached road network the features use.
    from osm_features import _CACHE_NAME, fetch_road_network  # noqa: PLC0415
    cache_path = project_root / "data" / "processed" / _CACHE_NAME
    try:
        payload = fetch_road_network((0, 0, 0, 0), cache_path)  # cache-only
    except Exception as exc:
        return {"feasible": False, "reason": f"road network unavailable ({exc})"}

    g = _build_local_graph(payload.get("ways", []), lat, lon, radius_km)
    if g.number_of_nodes() < 3:
        return {"feasible": False, "reason": "no mapped roads near the incident"}

    incident = (lat, lon)
    nearest, snap_m = _nearest_node(g, lat, lon)
    # The closure zone: every node within block_radius_m (≥ the nearest node).
    interior = {n for n in g.nodes if _haversine_m(incident, n) <= block_radius_m}
    interior.add(nearest)

    # Boundary exits: nodes just outside the zone that a closed road connects to.
    boundary: dict[tuple, float] = {}
    for u in interior:
        for v in g.neighbors(u):
            if v not in interior:
                boundary[v] = _bearing(incident, v)
    if len(boundary) < 2:
        return {"feasible": False,
                "reason": "no through-route around the closure (dead-end or single approach)",
                "blocked_point": [round(nearest[0], 5), round(nearest[1], 5)]}

    # entry/exit = the pair of exits most opposite across the closure (the through road).
    items = list(boundary.items())
    entry, exit_, best_gap = None, None, -1.0
    for i, (u, bu) in enumerate(items):
        for v, bv in items[i + 1:]:
            gap = _angle_gap(bu, bv)
            if gap > best_gap:
                entry, exit_, best_gap = u, v, gap

    import networkx as nx  # noqa: PLC0415
    h = g.copy()
    h.remove_nodes_from(interior)            # shut the closure zone; route around it
    try:
        path = nx.shortest_path(h, entry, exit_, weight="m")
        detour_m = nx.shortest_path_length(h, entry, exit_, weight="m")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return {"feasible": False, "reason": "no alternate route within the mapped area",
                "blocked_point": [round(nearest[0], 5), round(nearest[1], 5)]}

    # Reference "if open" distance = straight line between the two exits.
    direct_m = _haversine_m(entry, exit_)
    extra_m = detour_m - direct_m
    coords = [[round(n[0], 5), round(n[1], 5)] for n in path]
    return {
        "feasible": True,
        "blocked_point": [round(nearest[0], 5), round(nearest[1], 5)],
        "closure_radius_m": block_radius_m,
        "snap_distance_m": round(snap_m, 1),
        "direct_m": round(direct_m, 1),
        "detour_m": round(detour_m, 1),
        "extra_m": round(max(extra_m, 0.0), 1),
        "extra_pct": round(100 * extra_m / direct_m, 1) if direct_m > 0 else None,
        "n_segments": len(path) - 1,
        "detour_path": coords,
        "summary": (
            f"Reroute around a {block_radius_m:.0f} m closure: {detour_m:.0f} m detour "
            f"vs {direct_m:.0f} m straight-through (+{max(extra_m, 0):.0f} m)."
        ),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    import json
    # A point on the Outer Ring Road, Marathahalli — a busy arterial.
    print(json.dumps(plan_diversion(12.9568, 77.7011), indent=2))
