"""
impact_score.py — Transparent disruption-impact heuristic.

The dataset records NO measured congestion outcome (queue length, speed drop,
delay), so "event impact" cannot be fully *learned* from it. Rather than dress up
a guess as a model, we expose an explicit, documented composite of the disruption
drivers we have — now including the OSM road context (class + lanes), which is a
real, measured proxy for how much traffic a blockage actually affects:

    impact = w_closure · closure        + w_corridor · corridor_pressure
           + w_road    · road_capacity  + w_window   · high_incident_window

A closure on a 6-lane trunk road is genuinely more disruptive than one on a
residential lane, and the road term captures exactly that. It remains a HEURISTIC
prior, not a forecast — every term and weight is visible and the UI labels it as
such — but the road dimension grounds it in measured infrastructure rather than
time-of-day alone. A live-speed feed (roadmap) would turn this from a
capacity-weighted proxy into a directly *measured* impact.
"""
from __future__ import annotations

# Fixed, documented weights — closure dominates disruption, then corridor
# pressure, then the capacity of the affected road, then time-of-day load.
WEIGHTS = {"closure": 0.40, "corridor": 0.25, "road": 0.20, "window": 0.15}

# corridor_7d_score at/above which corridor pressure is treated as "saturated".
_CORRIDOR_SOFT_CAP = 30
# Lane count treated as a fully saturated arterial.
_LANES_SOFT_CAP = 6
# Neutral road factor when no road context is supplied ("unknown, assume mid").
_ROAD_NEUTRAL = 0.5


def _band(score: int) -> str:
    if score < 25:
        return "Low"
    if score < 50:
        return "Moderate"
    if score < 75:
        return "High"
    return "Severe"


def _road_factor(road_class_rank: int | None, lane_count: float | None) -> tuple[float, bool]:
    """
    0–1 capacity factor from OSM road class (rank 0–6) and lane count, or the
    neutral default when neither is known. Returns (factor, measured?).
    """
    if road_class_rank is None and lane_count is None:
        return _ROAD_NEUTRAL, False
    rank = max(0, min(int(road_class_rank or 0), 6)) / 6.0
    lanes = max(0.0, min(float(lane_count or 0), _LANES_SOFT_CAP)) / _LANES_SOFT_CAP
    # Class hierarchy carries most of the signal; lane count refines it.
    return 0.6 * rank + 0.4 * lanes, True


def impact_score(road_closure: bool = False, corridor_7d: float = 0.0,
                 is_peak: bool = False, closure_prob: float | None = None,
                 cluster_closure_rate: float | None = None,
                 road_class_rank: int | None = None,
                 lane_count: float | None = None) -> dict:
    """
    Return a 0–100 heuristic disruption-impact score with a full breakdown.

    closure factor: 1.0 if a closure is operator-flagged, else the model's
    P(closure), else the spatial cluster closure rate, else 0.
    road factor: capacity of the affected road (OSM class + lanes); neutral 0.5
    when the road context is unknown.
    """
    closure_factor = 1.0 if road_closure else float(
        closure_prob if closure_prob is not None
        else (cluster_closure_rate if cluster_closure_rate is not None else 0.0)
    )
    closure_factor = min(max(closure_factor, 0.0), 1.0)
    corridor_factor = min(max(float(corridor_7d), 0.0) / _CORRIDOR_SOFT_CAP, 1.0)
    road_factor, road_measured = _road_factor(road_class_rank, lane_count)
    window_factor = 1.0 if is_peak else 0.4

    raw = (WEIGHTS["closure"] * closure_factor
           + WEIGHTS["corridor"] * corridor_factor
           + WEIGHTS["road"] * road_factor
           + WEIGHTS["window"] * window_factor)
    score = int(round(100 * raw / sum(WEIGHTS.values())))

    return {
        "score": score,
        "label": _band(score),
        "breakdown": {
            "closure": round(closure_factor, 2),
            "corridor_pressure": round(corridor_factor, 2),
            "road_capacity": round(road_factor, 2),
            "high_incident_window": window_factor,
        },
        "weights": WEIGHTS,
        "road_measured": road_measured,
        "is_heuristic": True,
    }
