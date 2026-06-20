"""
resource_recommender.py — Rule-based traffic deployment planner.

Scoring table (from the SmartFlow plan)
----------------------------------------
Factor               Low   Medium   High
Base personnel         2      3       5
+peak hour            +0     +1      +1
+road_closure         +1     +1      +1
+high-risk cause      +1     +1      +1   (accident, flood, water_logging,
                                            protest, vip_movement, procession)

Barricade required   if road_closure OR cause in {accident, tree_fall, flood,
                     water_logging, protest, vip_movement, procession, public_event}
Diversion recommended if severity == High
Priority flag        URGENT  if High, PRIORITY if Medium, ROUTINE if Low
"""

from __future__ import annotations

import json
from pathlib import Path

from utils import get_nearest_station, is_peak_hour, load_config

# ---------------------------------------------------------------------------
# Rules — loaded from config.yaml (single source of truth), with fallbacks.
# ---------------------------------------------------------------------------

_DEFAULT_RULES = {
    "base_personnel": {"Low": 2, "Medium": 3, "High": 5},
    "peak_hour_bonus": 1,
    "road_closure_bonus": 1,
    "high_risk_cause_bonus": 1,
    # Crowd/gathering causes (protest, vip_movement, procession) need extra
    # officers for crowd control, not just clearance.
    "high_risk_causes": [
        "accident", "flood", "water_logging",
        "protest", "vip_movement", "procession",
    ],
    "barricade_causes": [
        "accident", "tree_fall", "flood", "water_logging",
        "protest", "vip_movement", "procession", "public_event",
    ],
    "diversion_min_severity": "High",
    # P(road closure) at/above which we recommend a barricade even if the
    # operator hasn't flagged a closure — driven by the calibrated model.
    "closure_prob_barricade_threshold": 0.30,
}


def _rules() -> dict:
    """Merge config.yaml resource_rules over the defaults."""
    rules = dict(_DEFAULT_RULES)
    try:
        cfg = load_config().get("resource_rules", {}) or {}
        for k, v in cfg.items():
            rules[k] = v
    except Exception:
        pass
    return rules


_PRIORITY_FLAGS = {"High": "URGENT", "Medium": "PRIORITY", "Low": "ROUTINE"}

# ---------------------------------------------------------------------------
# Empirical clearance ranges (cause → median + IQR), from clearance_stats.json.
# These are real historical close-times, NOT a model forecast.
# ---------------------------------------------------------------------------

_clearance_cache: dict | None = None


def _clearance_stats() -> dict:
    global _clearance_cache
    if _clearance_cache is None:
        path = Path(__file__).resolve().parents[1] / "data" / "processed" / "clearance_stats.json"
        try:
            _clearance_cache = json.loads(path.read_text())
        except Exception:
            _clearance_cache = {"_overall": {"median": 57, "p25": 30, "p75": 120}}
    return _clearance_cache


def clearance_range(cause: str) -> dict:
    """Return {median, p25, p75, n} for a cause, falling back to overall."""
    stats = _clearance_stats()
    return stats.get(cause, stats.get("_overall", {"median": 57, "p25": 30, "p75": 120}))


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
    closure_probability: float | None = None,
    barricade_threshold: float | None = None,
) -> dict:
    """
    Produce a deployment recommendation for a single event.

    Parameters
    ----------
    severity_class       : 'High' | 'Medium' | 'Low' (rules-based triage)
    event_cause          : canonical cause string
    requires_road_closure: bool (operator-flagged)
    hour_of_day          : 0–23
    zone                 : zone string from the dataset
    duration_minutes     : ignored for the headline estimate — kept for
                           backward-compat; clearance now uses empirical ranges
    closure_probability  : calibrated P(road closure) from the model (optional);
                           a high value triggers a barricade even if not flagged

    Returns
    -------
    dict with personnel_count, barricade_required, diversion_recommended,
         dispatch_from, clearance range, priority_flag, rationale
    """
    rules  = _rules()
    base_personnel = rules["base_personnel"]
    high_risk = set(rules["high_risk_causes"])
    barricade_causes = set(rules["barricade_causes"])

    sev    = severity_class if severity_class in base_personnel else "Low"
    cause  = event_cause if event_cause else "other"
    closure = bool(requires_road_closure)
    is_peak = is_peak_hour(hour_of_day)
    closure_prob = float(closure_probability) if closure_probability is not None else None
    # Prefer the model's cost-derived threshold (stamped in the closure pkl and
    # passed in); fall back to the config value if none supplied.
    thresh = barricade_threshold if barricade_threshold is not None \
        else rules["closure_prob_barricade_threshold"]
    closure_likely = closure_prob is not None and closure_prob >= thresh

    # ---------- Personnel ----------
    personnel = base_personnel[sev]
    personnel += rules["peak_hour_bonus"]   if is_peak and sev in {"Medium", "High"} else 0
    personnel += rules["road_closure_bonus"] if (closure or closure_likely)          else 0
    personnel += rules["high_risk_cause_bonus"] if cause in high_risk                else 0

    # ---------- Barricade ----------
    barricade = closure or closure_likely or (cause in barricade_causes)

    # ---------- Diversion ----------
    diversion = (sev == rules["diversion_min_severity"]) or closure_likely

    # ---------- Dispatch station ----------
    station = get_nearest_station(zone)

    # ---------- Clearance estimate (empirical range, NOT a model forecast) ----------
    cr = clearance_range(cause)
    clearance      = int(round(cr["median"]))
    clearance_low  = int(round(cr.get("p25", cr["median"])))
    clearance_high = int(round(cr.get("p75", cr["median"])))

    # ---------- Priority flag ----------
    flag = _PRIORITY_FLAGS[sev]

    # ---------- Human-readable rationale ----------
    rationale_parts = [f"{sev} severity event"]
    if is_peak:
        rationale_parts.append("high-incident window")
    if closure:
        rationale_parts.append("operator flagged road closure")
    elif closure_likely:
        rationale_parts.append(f"model: closure likely ({closure_prob*100:.0f}%)")
    if cause in high_risk:
        rationale_parts.append(f"high-risk cause ({cause})")

    return {
        "personnel_count":             personnel,
        "barricade_required":          barricade,
        "diversion_recommended":       diversion,
        "dispatch_from":               station,
        "estimated_clearance_minutes": clearance,
        "clearance_low":               clearance_low,
        "clearance_high":              clearance_high,
        "clearance_note":             f"typical close-time for {cause} (median of {cr.get('n','?')} past events)",
        "closure_probability":         closure_prob,
        "priority_flag":               flag,
        "is_peak_hour":                is_peak,
        "rationale":                   "; ".join(rationale_parts),
    }


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
