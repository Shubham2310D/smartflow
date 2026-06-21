"""
api/main.py — Thin real-time endpoint for SmartFlow.

The Streamlit dashboard is batch; this is the streaming entry point the brief
asks for. A single POST /event runs the exact same path the dashboard uses —
predict (severity triage + calibrated closure likelihood), look up the typical
clearance range, produce a deployment recommendation, and log the decision into
the learning loop — so a live feed and the UI stay consistent.

Run (after `pip install fastapi uvicorn`):
    uvicorn api.main:app --reload --port 8000
Then:
    curl -X POST localhost:8000/event -H "Content-Type: application/json" \
         -d '{"event_cause":"procession","zone":"Central Zone 1","hour_of_day":18,
              "description":"Ganesha procession near circle"}'
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel, Field, field_validator  # noqa: E402

from feature_engineering import (  # noqa: E402
    CAUSE_SEVERITY_WEIGHT, _SEMANTIC_MAP, _VEH_TYPE_MAP, _VEH_TYPE_ORDER,
    extract_semantic_type,
)
from model_training import SEVERITY_INVERSE_MAP, check_lib_versions  # noqa: E402
from outcomes_log import log_decision  # noqa: E402
from event_store import active_events, live_history, record_event  # noqa: E402
from history_features import history_features  # noqa: E402
from osm_features import road_context  # noqa: E402
from resource_recommender import clearance_range, recommend  # noqa: E402
from utils import is_peak_hour  # noqa: E402

app = FastAPI(title="SmartFlow API", version="1.0")

_MODELS = _ROOT / "models"
_clf = joblib.load(_MODELS / "severity_classifier.pkl")
_clo = joblib.load(_MODELS / "closure_predictor.pkl") if (_MODELS / "closure_predictor.pkl").exists() else None


class Event(BaseModel):
    # Validated inputs — out-of-range values are rejected with HTTP 422 rather
    # than silently producing a garbage prediction.
    event_cause: str = "vehicle_breakdown"
    zone: str = "Central Zone 1"
    corridor: str = ""
    hour_of_day: int = Field(9, ge=0, le=23)
    day_of_week: int = Field(0, ge=0, le=6)
    veh_type: str = "unknown"
    # Optional event location. When supplied, the event is snapped to its exact
    # OSM road (class + lanes); otherwise the corridor's typical road context is
    # used. Bounds are the Bengaluru operating area.
    latitude: float | None = Field(None, ge=12.5, le=13.5)
    longitude: float | None = Field(None, ge=77.2, le=77.9)
    # History features are looked up from the corridor's historical medians by
    # default (see history_features). Pass explicit values only to override —
    # e.g. when a real event store can supply a live count.
    junction_repeat_count: int | None = Field(None, ge=0)
    corridor_7d_score: int | None = Field(None, ge=0)
    description: str = Field("", max_length=2000)
    log: bool = True

    @field_validator("event_cause", "zone", "veh_type")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        v = (v or "").strip()
        return v if v else "unknown"


@app.get("/health")
def health():
    # Flag any model trained under a different sklearn/xgboost/numpy than the
    # runtime — version skew can silently break a pickled model.
    warns = check_lib_versions(_clf) + (check_lib_versions(_clo) if _clo else [])
    return {
        "status": "ok",
        "closure_model": _clo is not None,
        "model_versions": _clf.get("lib_versions"),
        "version_warnings": warns,
    }


@app.post("/event")
def handle_event(ev: Event):
    sem = extract_semantic_type(ev.description)

    # Backward-looking history features. Prefer REAL live state from the event
    # store once it's warm; otherwise the corridor-median lookup; an explicit
    # caller value always overrides. Never a fabricated constant.
    hist = history_features(ev.corridor or None)
    live = live_history(ev.corridor or None, project_root=_ROOT)   # {} if store empty
    jrc = (ev.junction_repeat_count if ev.junction_repeat_count is not None
           else live.get("junction_repeat_count", hist["junction_repeat_count"]))
    c7d = (ev.corridor_7d_score if ev.corridor_7d_score is not None
           else live.get("corridor_7d_score", hist["corridor_7d_score"]))

    # Road context: snap to the exact OSM road when a location is given, else use
    # the corridor's typical class / lane count from history.
    if ev.latitude is not None and ev.longitude is not None:
        rc = road_context(ev.latitude, ev.longitude, project_root=_ROOT)
        road_rank, lane_cnt, road_src = rc["road_class_rank"], rc["lane_count"], "osm_snap"
    else:
        road_rank = hist.get("road_class_rank", 0)
        lane_cnt = hist.get("lane_count", 2)
        road_src = "corridor_history"

    feats = {
        "cause_severity_weight":  CAUSE_SEVERITY_WEIGHT.get(ev.event_cause, 1),
        "road_closure_binary":    0,
        "event_semantic_encoded": _SEMANTIC_MAP.get(sem, 0),
        "hour_of_day":            ev.hour_of_day,
        "day_of_week":            ev.day_of_week,
        "is_peak_hour":           int(is_peak_hour(ev.hour_of_day)),
        "is_weekend":             int(ev.day_of_week >= 5),
        "junction_repeat_count":  jrc,
        "corridor_7d_score":      c7d,
        "cluster_prior_events":   hist["cluster_prior_events"],
        "cluster_closure_rate":   hist["cluster_closure_rate"],
        "road_class_rank":        road_rank,
        "lane_count":             lane_cnt,
        "veh_type_encoded":       _VEH_TYPE_MAP.get(ev.veh_type, len(_VEH_TYPE_ORDER) - 1),
    }
    X_clf = pd.DataFrame([feats])[_clf["features"]]
    sev = SEVERITY_INVERSE_MAP[int(_clf["model"].predict_proba(X_clf)[0].argmax())]

    closure_prob = None
    if _clo is not None:
        X_clo = pd.DataFrame([feats])[_clo["features"]]
        closure_prob = float(_clo["model"].predict_proba(X_clo)[0][1])

    rec = recommend(
        severity_class=sev, event_cause=ev.event_cause, requires_road_closure=False,
        hour_of_day=ev.hour_of_day, zone=ev.zone, closure_probability=closure_prob,
        barricade_threshold=(_clo.get("barricade_threshold") if _clo else None),
    )

    # If a diversion is warranted AND we know where the event is, compute a REAL
    # reroute around the blockage (not just the boolean flag).
    diversion_plan = None
    if rec.get("diversion_recommended") and ev.latitude is not None and ev.longitude is not None:
        from diversion import plan_diversion  # noqa: PLC0415 — lazy: only when needed
        diversion_plan = plan_diversion(ev.latitude, ev.longitude, project_root=_ROOT)

    if ev.log:
        log_decision({
            "event_cause": ev.event_cause, "zone": ev.zone, "hour_of_day": ev.hour_of_day,
            "predicted_severity": sev, "predicted_clearance_min": rec["estimated_clearance_minutes"],
            "recommended_personnel": rec["personnel_count"],
        }, project_root=_ROOT)
        # Record into the live event store so subsequent events get REAL history.
        record_event({
            "event_cause": ev.event_cause, "zone": ev.zone, "corridor": ev.corridor or None,
            "severity": sev, "closure_prob": closure_prob, "status": "active",
        }, project_root=_ROOT)

    return {
        "severity_triage": sev,
        "detected_event_type": sem,
        "road_closure_likelihood": closure_prob,
        "clearance_range": clearance_range(ev.event_cause),
        "recommendation": rec,
        "diversion_plan": diversion_plan,   # real reroute when a location is given
        # Echo the history features actually used, so the caller can see they
        # came from corridor history (or an override), not a fabricated constant.
        "history_features_used": {
            "corridor": ev.corridor or None,
            "junction_repeat_count": jrc,
            "corridor_7d_score": c7d,
            "source": ("override" if (ev.junction_repeat_count is not None or ev.corridor_7d_score is not None)
                       else "live_store" if live else "corridor_history"),
        },
        "road_context_used": {
            "road_class_rank": road_rank,
            "lane_count": lane_cnt,
            "source": road_src,
        },
    }


@app.get("/active")
def active(limit: int = 100):
    """Currently-active events from the live store (for an operations console)."""
    df = active_events(project_root=_ROOT, limit=limit)
    return {"count": int(len(df)), "events": df.to_dict("records")}
