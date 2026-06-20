"""
event_analog.py — Case-based recommender for planned/known events.

Planned events (processions, VIP movement, protests, public events) have a known
type and location ahead of time — they are the *forecastable* half of the brief.
Rather than a model, we answer them the way an experienced controller would:
"the last N events like this, here, needed roughly this much."

Given a cause + locality (zone, or lat/lon), we retrieve the most similar past
events and summarise what actually happened — how long they took to clear and how
often they required a road closure — plus a personnel estimate from the same
rule engine. No training, no leakage, directly grounded in history.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# The planned / gathering event types this module is meant for
PLANNED_CAUSES = ["procession", "vip_movement", "protest", "public_event"]

_feats_cache: pd.DataFrame | None = None


def _load_features(project_root: Path | None = None) -> pd.DataFrame:
    global _feats_cache
    if _feats_cache is None:
        if project_root is None:
            project_root = Path(__file__).resolve().parents[1]
        path = project_root / "data" / "processed" / "features.csv"
        cols = ["event_cause", "latitude", "longitude", "zone", "address",
                "duration_minutes", "road_closure_binary", "severity_class",
                "start_datetime"]
        df = pd.read_csv(path, usecols=lambda c: c in cols, parse_dates=["start_datetime"])
        _feats_cache = df
    return _feats_cache


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def find_analogs(
    cause: str,
    zone: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    k: int = 10,
    project_root: Path | None = None,
) -> dict:
    """
    Retrieve the k most similar past events of `cause` and summarise outcomes.

    Locality preference: by coordinates if given (nearest k), else by zone, else
    citywide for that cause.  Returns medians/rates plus a small sample table.
    """
    df = _load_features(project_root)
    all_cause = df[df["event_cause"] == cause].copy()
    scope = "citywide"
    pool = all_cause

    _MIN_LOCAL = 3   # below this, a locality estimate is too noisy — go citywide

    if lat is not None and lon is not None and len(all_cause):
        located = all_cause.dropna(subset=["latitude", "longitude"]).copy()
        located["distance_km"] = _haversine_km(lat, lon, located["latitude"], located["longitude"])
        pool = located.sort_values("distance_km").head(k)
        scope = "nearest by location"
    elif zone and (all_cause["zone"] == zone).sum() >= _MIN_LOCAL:
        pool = all_cause[all_cause["zone"] == zone].head(k)
        scope = f"in {zone}"
    else:
        # Too few in this zone for a stable estimate → use all events of this type.
        pool = all_cause.head(max(k, 25))
        scope = "citywide (too few local)" if zone else "citywide"

    if pool.empty:
        return {"found": 0, "cause": cause, "scope": scope}

    dur = pool["duration_minutes"].dropna()
    return {
        "found":            int(len(pool)),
        "cause":            cause,
        "scope":            scope,
        "median_clearance": round(float(dur.median()), 0) if len(dur) else None,
        "p25_clearance":    round(float(dur.quantile(0.25)), 0) if len(dur) else None,
        "p75_clearance":    round(float(dur.quantile(0.75)), 0) if len(dur) else None,
        "closure_rate":     round(float(pool["road_closure_binary"].mean()) * 100, 0)
                            if "road_closure_binary" in pool else None,
        "samples":          pool[[
            c for c in ["start_datetime", "address", "zone", "duration_minutes",
                        "road_closure_binary", "distance_km"] if c in pool.columns
        ]].head(8),
    }
