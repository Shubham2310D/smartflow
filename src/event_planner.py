"""
event_planner.py — Advance plan for a known/upcoming event.

The forecastable half of the brief: planned events (processions, VIP movement,
protests, public events, construction) have a known type, place and time *before*
they happen. This module produces a single forward-looking plan from that input,
combining the two honest pieces we already have:

  1. case-based IMPACT forecast — what similar past events actually required
     (expected severity, road-closure rate, clearance range), via event_analog
     with graceful backoff + a confidence rating; and
  2. a DEPLOYMENT plan — personnel, barricade, diversion, dispatch station — from
     the rule engine, seeded with the forecast above (the historical closure rate
     becomes the closure-likelihood that drives barricading).

No model is invented for the ~130 gathering rows the data can't support; the plan
is grounded in retrieval and labelled with how much evidence backs it.
"""
from __future__ import annotations

from pathlib import Path

from event_analog import PLANNED_OBSTRUCTIONS, find_analogs
from resource_recommender import recommend


def plan_event(
    cause: str,
    zone: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    hour: int = 12,
    k: int = 10,
    project_root: Path | None = None,
) -> dict:
    """
    Build an advance plan for an upcoming event of `cause` at `zone`/coords/`hour`.

    Returns the impact forecast (from analogs), the expected severity it implies,
    and the deployment recommendation seeded with that forecast.
    """
    analog = find_analogs(cause, zone=zone, lat=lat, lon=lon, k=k, project_root=project_root)

    severity = analog.get("expected_severity") or "Medium"
    # Historical road-closure rate for this kind of event → closure likelihood
    # that drives the barricade decision in the rule engine.
    closure_rate = analog.get("closure_rate")
    closure_prob = (closure_rate / 100.0) if closure_rate is not None else None

    rec = recommend(
        severity_class=severity,
        event_cause=cause,
        requires_road_closure=False,            # not operator-flagged; inferred below
        hour_of_day=hour,
        zone=zone or "Unknown",
        closure_probability=closure_prob,
    )

    return {
        "cause":              cause,
        "is_obstruction":     cause in PLANNED_OBSTRUCTIONS,
        "expected_severity":  severity,
        "confidence":         analog.get("confidence", "none"),
        "closure_likelihood": closure_prob,     # 0–1, historical rate for this type
        "analog":             analog,
        "recommendation":     rec,
    }
