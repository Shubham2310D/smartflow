"""
impact_score.py — Transparent disruption-impact heuristic.

The dataset records NO measured congestion outcome (queue length, speed drop,
delay), so "event impact" cannot be *learned* from it. Rather than dress up a
guess as a model, we expose an explicit, documented composite of the disruption
drivers we do have:

    impact = w_closure · closure + w_corridor · corridor_pressure
             + w_window · high_incident_window          (weights below)

It is a HEURISTIC prior, not a forecast — every term and weight is visible and
the UI labels it as such. The OSM road-class / lane-count (or a typical-speed
feed) join on the roadmap is what would turn this into a *measured* impact; until
then this is the honest stand-in the brief's "forecast impact" asks for.
"""
from __future__ import annotations

# Fixed, documented weights — closure dominates disruption, then corridor
# pressure, then time-of-day load.
WEIGHTS = {"closure": 0.5, "corridor": 0.3, "window": 0.2}

# corridor_7d_score at/above which corridor pressure is treated as "saturated".
_CORRIDOR_SOFT_CAP = 30


def _band(score: int) -> str:
    if score < 25:
        return "Low"
    if score < 50:
        return "Moderate"
    if score < 75:
        return "High"
    return "Severe"


def impact_score(road_closure: bool = False, corridor_7d: float = 0.0,
                 is_peak: bool = False, closure_prob: float | None = None,
                 cluster_closure_rate: float | None = None) -> dict:
    """
    Return a 0–100 heuristic disruption-impact score with a full breakdown.

    closure factor: 1.0 if a closure is operator-flagged, else the model's
    P(closure), else the spatial cluster closure rate, else 0.
    """
    closure_factor = 1.0 if road_closure else float(
        closure_prob if closure_prob is not None
        else (cluster_closure_rate if cluster_closure_rate is not None else 0.0)
    )
    closure_factor = min(max(closure_factor, 0.0), 1.0)
    corridor_factor = min(max(float(corridor_7d), 0.0) / _CORRIDOR_SOFT_CAP, 1.0)
    window_factor = 1.0 if is_peak else 0.4

    raw = (WEIGHTS["closure"] * closure_factor
           + WEIGHTS["corridor"] * corridor_factor
           + WEIGHTS["window"] * window_factor)
    score = int(round(100 * raw / sum(WEIGHTS.values())))

    return {
        "score": score,
        "label": _band(score),
        "breakdown": {
            "closure": round(closure_factor, 2),
            "corridor_pressure": round(corridor_factor, 2),
            "high_incident_window": window_factor,
        },
        "weights": WEIGHTS,
        "is_heuristic": True,
    }
