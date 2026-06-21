"""
osm_features.py — Join each incident to the road it happened on (OpenStreetMap).

The raw dataset has lat/lon + timestamp but *no road context*: a closure on a
6-lane arterial and one on a residential lane look identical to the models. This
module fixes that — the single highest-leverage external join in the roadmap.

What it does
------------
1. One Overpass query for the drivable road network inside the incident bounding
   box (motorway → residential), cached to data/processed/osm_roads.json so the
   network call happens once and everything downstream is offline/reproducible.
2. For every incident, nearest-road assignment (KD-tree over road geometry
   sample points, equirectangular projection — accurate at city scale):
       road_class       — OSM highway class of the nearest road (str)
       road_class_rank  — ordinal 0–6 (residential … motorway); the model feature
       lane_count       — OSM `lanes` tag, else a class-based default
       road_dist_m      — metres to that road (a match-quality / trust signal)

Design choices
--------------
* Offline-cacheable: the brief values reproducibility; once osm_roads.json is
  committed, no network is needed to rebuild features or train.
* Graceful degradation: if Overpass is unreachable AND no cache exists, every
  incident gets road_class_rank=0 / class-default lanes and a logged warning —
  the pipeline never hard-fails on a network blip.
* No new heavy deps: uses scipy.spatial.cKDTree (already required) — not
  geopandas/osmnx.

This module is also the backbone for src/diversion.py (real reroute) — both read
the same cached road graph.
"""

from __future__ import annotations

import gzip
import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical on-disk cache: a SLIM, gzipped road network (≈5 MB vs ~73 MB raw
# Overpass), holding only what features + routing need — committed for offline
# rebuilds. Schema: {"ways": [{"h": highway_class, "l": lanes_or_null,
# "g": [[lat, lon], …]}, …]}.
_CACHE_NAME = "osm_roads.json.gz"

# ---------------------------------------------------------------------------
# Road-class hierarchy → ordinal rank (higher = bigger road = more disruption
# when blocked). Tracks the OSM highway key. _link ramps fold to their parent.
# ---------------------------------------------------------------------------
ROAD_CLASS_RANK: dict[str, int] = {
    "motorway": 6, "motorway_link": 6,
    "trunk": 5, "trunk_link": 5,
    "primary": 4, "primary_link": 4,
    "secondary": 3, "secondary_link": 3,
    "tertiary": 2, "tertiary_link": 2,
    "unclassified": 1, "residential": 1, "living_street": 1, "service": 1,
}
# Classes we ask Overpass for (anything finer than residential is noise here).
_QUERY_CLASSES = (
    "motorway|trunk|primary|secondary|tertiary|unclassified|residential"
    "|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link"
)
# Typical lane count when the OSM `lanes` tag is missing (it usually is on
# smaller roads). Conservative, class-based.
_LANES_DEFAULT_BY_RANK = {6: 6, 5: 4, 4: 4, 3: 3, 2: 2, 1: 2, 0: 1}

_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
_USER_AGENT = "SmartFlow/1.0 (Bengaluru traffic hackathon; research use)"
# Beyond this distance an incident isn't really "on" the queried network, so we
# treat it as a minor/unknown road rather than snapping to a far-off arterial.
_MAX_SNAP_M = 120.0
_EARTH_R_M = 6_371_000.0


# ---------------------------------------------------------------------------
# 1. Fetch + cache the road network
# ---------------------------------------------------------------------------

def _bbox(df: pd.DataFrame, pad: float = 0.01) -> tuple[float, float, float, float]:
    """(south, west, north, east) padded slightly past the incident extent."""
    return (
        float(df["latitude"].min()) - pad,
        float(df["longitude"].min()) - pad,
        float(df["latitude"].max()) + pad,
        float(df["longitude"].max()) + pad,
    )


def _slim(payload: dict) -> dict:
    """Reduce a raw Overpass response to the committed slim schema (see _CACHE_NAME)."""
    ways = []
    for el in payload.get("elements", []):
        if el.get("type") != "way":
            continue
        tags = el.get("tags", {})
        geom = [[round(p["lat"], 5), round(p["lon"], 5)] for p in el.get("geometry", [])]
        if not geom:
            continue
        ways.append({"h": tags.get("highway", ""), "l": tags.get("lanes"), "g": geom})
    return {"ways": ways}


def fetch_road_network(bbox: tuple[float, float, float, float],
                       cache_path: Path, force: bool = False,
                       timeout: int = 180) -> dict:
    """
    Return the slim road network for `bbox`, using the gzipped cache when present.

    The cache makes this deterministic and offline after the first successful
    run. Raises only if there is neither a cache nor a reachable Overpass.
    """
    if cache_path.exists() and not force:
        logger.info("OSM road network: using cache %s", cache_path.name)
        with gzip.open(cache_path, "rt", encoding="utf-8") as f:
            return json.load(f)

    s, w, n, e = bbox
    query = (
        f"[out:json][timeout:{timeout}];"
        f'way["highway"~"^({_QUERY_CLASSES})$"]({s},{w},{n},{e});'
        f"out tags geom;"
    )
    data = urllib.parse.urlencode({"data": query}).encode()
    req = urllib.request.Request(
        _OVERPASS_URL, data=data,
        headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
    )
    logger.info("OSM road network: querying Overpass for bbox %s …", bbox)
    with urllib.request.urlopen(req, timeout=timeout + 30) as resp:
        payload = json.load(resp)

    slim = _slim(payload)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_path, "wt", encoding="utf-8") as f:
        json.dump(slim, f, separators=(",", ":"))
    logger.info("OSM road network: cached %d ways → %s",
                len(slim["ways"]), cache_path.name)
    return slim


# ---------------------------------------------------------------------------
# 2. Parse ways → sample points for nearest-road lookup
# ---------------------------------------------------------------------------

def _parse_lanes(tags: dict) -> float | None:
    raw = tags.get("lanes")
    if raw is None:
        return None
    try:
        # "2", "3", or sometimes "2;3" (forward;backward) — take the max.
        return float(max(int(float(x)) for x in str(raw).replace(",", ";").split(";")))
    except (ValueError, TypeError):
        return None


def _build_point_index(ways: list[dict]):
    """
    Flatten every road way into its constituent geometry points, each tagged with
    that road's class rank and lane count. Reads the slim schema (see _CACHE_NAME).
    Returns:
        pts  : (N, 2) array of [lat, lon]
        rank : (N,)   int8  road_class_rank per point
        lanes: (N,)   float lane count per point (NaN if unknown)
        klass: (N,)   object road_class string per point
    """
    lats, lons, ranks, lanes, klass = [], [], [], [], []
    for w in ways:
        hw = w.get("h", "")
        rank = ROAD_CLASS_RANK.get(hw, 0)
        ln = _parse_lanes({"lanes": w.get("l")})
        for lat, lon in w.get("g", []):
            lats.append(lat)
            lons.append(lon)
            ranks.append(rank)
            lanes.append(ln if ln is not None else np.nan)
            klass.append(hw)
    if not lats:
        return None
    return (
        np.column_stack([np.array(lats), np.array(lons)]),
        np.array(ranks, dtype=np.int8),
        np.array(lanes, dtype=float),
        np.array(klass, dtype=object),
    )


def _project(latlon: np.ndarray, lat0: float) -> np.ndarray:
    """Equirectangular metres relative to lat0 — exact enough for nearest-road."""
    lat_m = np.radians(latlon[:, 0]) * _EARTH_R_M
    lon_m = np.radians(latlon[:, 1]) * _EARTH_R_M * np.cos(np.radians(lat0))
    return np.column_stack([lat_m, lon_m])


# Shared, lazily-built spatial index so both the batch join (add_road_features)
# and per-request lookups (road_context, used by the API) reuse one KD-tree
# instead of rebuilding it. Keyed by project root.
_INDEX_CACHE: dict[str, dict] = {}


def _get_index(project_root: Path) -> dict | None:
    """Build (once) and return the road KD-tree + per-point class/lane arrays."""
    from scipy.spatial import cKDTree  # noqa: PLC0415

    key = str(project_root)
    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key]

    cache_path = project_root / "data" / "processed" / _CACHE_NAME
    try:
        payload = fetch_road_network((0, 0, 0, 0), cache_path)  # cache-only here
    except Exception as exc:
        logger.warning("road index unavailable (%s)", exc)
        return None
    idx = _build_point_index(payload.get("ways", []))
    if idx is None:
        return None
    pts, ranks, lanes, klass = idx
    lat0 = float(pts[:, 0].mean())   # fixed projection origin → reusable for any point
    entry = {"tree": cKDTree(_project(pts, lat0)), "ranks": ranks,
             "lanes": lanes, "klass": klass, "lat0": lat0}
    _INDEX_CACHE[key] = entry
    return entry


def road_context(lat: float, lon: float, project_root: Path | None = None) -> dict:
    """
    Snap a single (lat, lon) to its nearest road. Returns
    {road_class, road_class_rank, lane_count, road_dist_m}, with safe minor-road
    defaults if the point is too far from any road or the network is unavailable.
    Used by the real-time API, where each event carries a real location.
    """
    default = {"road_class": "minor_or_unknown", "road_class_rank": 0,
               "lane_count": float(_LANES_DEFAULT_BY_RANK[0]), "road_dist_m": None}
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]
    if lat is None or lon is None:
        return default
    entry = _get_index(project_root)
    if entry is None:
        return default

    dist_m, nn = entry["tree"].query(_project(np.array([[lat, lon]]), entry["lat0"]), k=1)
    d = float(np.atleast_1d(dist_m)[0]); j = int(np.atleast_1d(nn)[0])
    if d > _MAX_SNAP_M:
        return {**default, "road_dist_m": round(d, 1)}
    rank = int(entry["ranks"][j])
    ln = entry["lanes"][j]
    if np.isnan(ln):
        ln = _LANES_DEFAULT_BY_RANK[rank]
    return {"road_class": str(entry["klass"][j]), "road_class_rank": rank,
            "lane_count": float(ln), "road_dist_m": round(d, 1)}


# ---------------------------------------------------------------------------
# 3. Public entry point — assign road features to a DataFrame
# ---------------------------------------------------------------------------

def add_road_features(df: pd.DataFrame, project_root: Path | None = None,
                      force_refresh: bool = False) -> pd.DataFrame:
    """
    Add road_class / road_class_rank / lane_count / road_dist_m to `df`.

    Requires latitude/longitude. Degrades gracefully (rank 0, class-default
    lanes) for points with no nearby road or when the network is unavailable.
    """
    from scipy.spatial import cKDTree  # noqa: PLC0415

    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]
    cache_path = project_root / "data" / "processed" / _CACHE_NAME

    df = df.copy()
    n = len(df)
    # Defaults applied when the join can't run or a point is too far from a road.
    df["road_class"] = "minor_or_unknown"
    df["road_class_rank"] = 0
    df["lane_count"] = _LANES_DEFAULT_BY_RANK[0]
    df["road_dist_m"] = np.nan

    coords_mask = df["latitude"].notna() & df["longitude"].notna()
    if not coords_mask.any():
        logger.warning("add_road_features: no valid coordinates — defaults applied")
        return df

    try:
        payload = fetch_road_network(_bbox(df[coords_mask]), cache_path,
                                     force=force_refresh)
    except Exception as exc:
        logger.warning(
            "add_road_features: road network unavailable (%s) — defaults applied. "
            "Commit data/processed/osm_roads.json to make this offline.", exc)
        return df

    idx = _build_point_index(payload.get("ways", []))
    if idx is None:
        logger.warning("add_road_features: road network empty — defaults applied")
        return df
    pts, ranks, lanes, klass = idx

    lat0 = float(df.loc[coords_mask, "latitude"].mean())
    tree = cKDTree(_project(pts, lat0))
    inc = df.loc[coords_mask, ["latitude", "longitude"]].to_numpy(dtype=float)
    dist_m, nn = tree.query(_project(inc, lat0), k=1)

    near = dist_m <= _MAX_SNAP_M           # only snap when genuinely on a road
    nn_rank = ranks[nn]
    nn_lanes = lanes[nn]
    nn_class = klass[nn]

    # Lane count: OSM tag if present, else class-based default.
    rank_default = np.array([_LANES_DEFAULT_BY_RANK[r] for r in nn_rank], dtype=float)
    nn_lanes = np.where(np.isnan(nn_lanes), rank_default, nn_lanes)

    pos = np.flatnonzero(coords_mask.to_numpy())
    snapped = pos[near]
    df.iloc[snapped, df.columns.get_loc("road_class")] = nn_class[near]
    df.iloc[snapped, df.columns.get_loc("road_class_rank")] = nn_rank[near].astype(int)
    df.iloc[snapped, df.columns.get_loc("lane_count")] = nn_lanes[near]
    df.iloc[pos, df.columns.get_loc("road_dist_m")] = np.round(dist_m, 1)

    matched = int(near.sum())
    logger.info(
        "add_road_features: %d/%d incidents snapped to a road (≤%.0fm); "
        "mean lanes=%.1f, mean class_rank=%.2f",
        matched, n, _MAX_SNAP_M, float(df["lane_count"].mean()),
        float(df["road_class_rank"].mean()),
    )
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    root = Path(__file__).resolve().parents[1]
    clean = pd.read_csv(root / "data" / "processed" / "clean.csv")
    out = add_road_features(clean, project_root=root)
    print(out[["road_class", "road_class_rank", "lane_count", "road_dist_m"]].describe(include="all"))
    print(out["road_class"].value_counts())
