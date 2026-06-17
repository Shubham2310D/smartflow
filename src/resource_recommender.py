"""
resource_recommender.py — Rule-based traffic deployment planner.

Scoring table (from the SmartFlow plan)
----------------------------------------
Factor               Low   Medium   High
Base personnel         2      3       5
+peak hour            +0     +1      +1
+road_closure         +1     +1      +1
+high-risk cause      +1     +1      +1   (accident, flood, water_logging)

Barricade required   if road_closure OR cause in {accident, tree_fall, flood, water_logging}
Diversion recommended if severity == High
Priority flag        URGENT  if High, PRIORITY if Medium, ROUTINE if Low
"""

from __future__ import annotations

from utils import get_nearest_station

# ---------------------------------------------------------------------------
# Scoring tables
# ---------------------------------------------------------------------------

_BASE_PERSONNEL = {"Low": 2, "Medium": 3, "High": 5}

_HIGH_RISK_CAUSES = {"accident", "flood", "water_logging"}

_BARRICADE_CAUSES = {"accident", "tree_fall", "flood", "water_logging"}

_PRIORITY_FLAGS = {"High": "URGENT", "Medium": "PRIORITY", "Low": "ROUTINE"}

_DIVERSION_MIN_SEVERITY = {"High"}

# Estimated baseline clearance time by cause (minutes)
_CAUSE_BASE_CLEARANCE: dict[str, int] = {
    "accident":        55,
    "flood":           90,
    "water_logging":   70,
    "tree_fall":       45,
    "construction":    60,
    "public_event":    120,
    "pot_holes":       30,
    "vehicle_breakdown": 40,
    "other":           35,
}

_SEVERITY_CLEARANCE_MULTIPLIER = {"Low": 0.7, "Medium": 1.0, "High": 1.4}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def recommend(
    severity_class: str,
    event_cause: str,
    requires_road_closure: bool,
    hour_of_day: int,
    zone: str,
    duration_minutes: float | None = None,
) -> dict:
    """
    Produce a deployment recommendation for a single event.

    Parameters
    ----------
    severity_class       : 'High' | 'Medium' | 'Low'
    event_cause          : canonical cause string from CAUSE_SEVERITY_WEIGHT
    requires_road_closure: bool
    hour_of_day          : 0–23
    zone                 : zone string from the dataset
    duration_minutes     : predicted duration from the ML model (optional)

    Returns
    -------
    dict with personnel_count, barricade_required, diversion_recommended,
         dispatch_from, estimated_clearance_minutes, priority_flag, rationale
    """
    sev    = severity_class if severity_class in _BASE_PERSONNEL else "Low"
    cause  = event_cause if event_cause else "other"
    closure = bool(requires_road_closure)
    is_peak = _is_peak_hour(hour_of_day)

    # ---------- Personnel ----------
    personnel = _BASE_PERSONNEL[sev]
    personnel += 1 if is_peak and sev in {"Medium", "High"}   else 0
    personnel += 1 if closure                                  else 0
    personnel += 1 if cause in _HIGH_RISK_CAUSES               else 0

    # ---------- Barricade ----------
    barricade = closure or (cause in _BARRICADE_CAUSES)

    # ---------- Diversion ----------
    diversion = sev in _DIVERSION_MIN_SEVERITY

    # ---------- Dispatch station ----------
    station = get_nearest_station(zone)

    # ---------- Clearance estimate ----------
    if duration_minutes and duration_minutes > 0:
        clearance = int(round(duration_minutes))
    else:
        base = _CAUSE_BASE_CLEARANCE.get(cause, 40)
        mult = _SEVERITY_CLEARANCE_MULTIPLIER[sev]
        clearance = int(round(base * mult))
        if is_peak:
            clearance = int(round(clearance * 1.2))

    # ---------- Priority flag ----------
    flag = _PRIORITY_FLAGS[sev]

    # ---------- Human-readable rationale ----------
    rationale_parts = [f"{sev} severity event"]
    if is_peak:
        rationale_parts.append("peak-hour traffic")
    if closure:
        rationale_parts.append("road closure required")
    if cause in _HIGH_RISK_CAUSES:
        rationale_parts.append(f"high-risk cause ({cause})")

    return {
        "personnel_count":           personnel,
        "barricade_required":        barricade,
        "diversion_recommended":     diversion,
        "dispatch_from":             station,
        "estimated_clearance_minutes": clearance,
        "priority_flag":             flag,
        "is_peak_hour":              is_peak,
        "rationale":                 "; ".join(rationale_parts),
    }


def _is_peak_hour(hour: int) -> bool:
    return hour in range(8, 11) or hour in range(17, 21)


# ---------------------------------------------------------------------------
# Batch helper (for analytics / what-if scenarios)
# ---------------------------------------------------------------------------

def batch_recommend(df) -> "pd.DataFrame":
    """Apply recommend() row-wise to a DataFrame; returns DataFrame of results."""
    import pandas as pd
    records = []
    for _, row in df.iterrows():
        rec = recommend(
            severity_class        = row.get("severity_class", "Low"),
            event_cause           = row.get("event_cause", "other"),
            requires_road_closure = bool(row.get("road_closure_binary", 0)),
            hour_of_day           = int(row.get("hour_of_day", 12)),
            zone                  = row.get("zone", "Unknown"),
            duration_minutes      = row.get("duration_minutes"),
        )
        records.append(rec)
    return pd.DataFrame(records)


if __name__ == "__main__":
    # Quick smoke test
    result = recommend(
        severity_class="High",
        event_cause="accident",
        requires_road_closure=True,
        hour_of_day=18,
        zone="North Zone 1",
        duration_minutes=52.0,
    )
    for k, v in result.items():
        print(f"  {k:<35} {v}")
