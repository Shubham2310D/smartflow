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

# The planned / gathering event types this module is meant for (crowds).
PLANNED_CAUSES = ["procession", "vip_movement", "protest", "public_event"]

# Construction is plannable too, but it is a *sustained obstruction* with no
# crowd — different operationally (no crowd-control personnel, longer footprint),
# so it is kept as its own sub-case rather than folded in with gatherings.
PLANNED_OBSTRUCTIONS = ["construction"]

# Everything the event planner can forecast ahead of time.
PLANNABLE_EVENTS = PLANNED_CAUSES + PLANNED_OBSTRUCTIONS

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


_MIN_LOCAL = 3   # below this, a locality estimate is too noisy — back off


def _confidence(found: int, local: bool) -> str:
    """Confidence in the estimate from how many analogs matched and how local."""
    if found == 0:
        return "none"
    if found >= 8 and local:
        return "high"
    if found >= 8 or (found >= 3 and local):
        return "medium"
    return "low"


def find_analogs(
    cause: str,
    zone: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    k: int = 10,
    project_root: Path | None = None,
) -> dict:
    """
    Retrieve the most similar past events and summarise outcomes, with graceful
    backoff and an explicit confidence so a thin match reads as "low confidence"
    rather than a confident answer built on one data point.

    Backoff tiers (most → least specific):
      1. nearby            — same cause, nearest by coordinates (if lat/lon given)
      2. type_in_zone      — same cause, same zone (needs >= _MIN_LOCAL)
      3. type_citywide     — same cause, anywhere
      4. similar_citywide  — any planned event (only if this cause is near-absent)
    """
    df = _load_features(project_root)
    all_cause = df[df["event_cause"] == cause].copy()

    local = False
    if lat is not None and lon is not None and len(all_cause):
        located = all_cause.dropna(subset=["latitude", "longitude"]).copy()
        located["distance_km"] = _haversine_km(lat, lon, located["latitude"], located["longitude"])
        pool = located.sort_values("distance_km").head(k)
        tier, scope, local = "nearby", "nearest by location", True
    elif zone and (all_cause["zone"] == zone).sum() >= _MIN_LOCAL:
        pool = all_cause[all_cause["zone"] == zone].head(k)
        tier, scope, local = "type_in_zone", f"in {zone}", True
    elif len(all_cause) >= _MIN_LOCAL:
        pool = all_cause.head(max(k, 25))
        tier = "type_citywide"
        scope = "citywide (too few local)" if zone else "citywide"
    else:
        # This cause is near-absent — fall back to the broader planned-event pool
        # so the planner still has *some* grounded evidence, flagged as such.
        pool = df[df["event_cause"].isin(PLANNABLE_EVENTS)].head(max(k, 25))
        tier, scope = "similar_citywide", "all planned events (this type is rare)"

    if pool.empty:
        return {"found": 0, "cause": cause, "scope": scope, "tier": "none",
                "confidence": "none", "expected_severity": None}

    found = int(len(pool))
    dur = pool["duration_minutes"].dropna()
    sev = pool["severity_class"].dropna() if "severity_class" in pool else pd.Series(dtype=str)
    return {
        "found":            found,
        "cause":            cause,
        "scope":            scope,
        "tier":             tier,
        "confidence":       _confidence(found, local),
        "expected_severity": str(sev.mode().iloc[0]) if len(sev) else "Medium",
        "median_clearance": round(float(dur.median()), 0) if len(dur) else None,
        "p25_clearance":    round(float(dur.quantile(0.25)), 0) if len(dur) else None,
        "p75_clearance":    round(float(dur.quantile(0.75)), 0) if len(dur) else None,
        "closure_rate":     round(float(pool["road_closure_binary"].mean()) * 100, 0)
                            if "road_closure_binary" in pool else None,
        "samples":          pool[[
            c for c in ["start_datetime", "address", "zone", "severity_class",
                        "duration_minutes", "road_closure_binary", "distance_km"]
            if c in pool.columns
        ]].head(8),
    }
