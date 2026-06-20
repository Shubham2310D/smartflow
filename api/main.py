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
from pydantic import BaseModel  # noqa: E402

from feature_engineering import (  # noqa: E402
    CAUSE_SEVERITY_WEIGHT, _SEMANTIC_MAP, _VEH_TYPE_MAP, _VEH_TYPE_ORDER,
    extract_semantic_type,
)
from model_training import SEVERITY_INVERSE_MAP  # noqa: E402
from outcomes_log import log_decision  # noqa: E402
from resource_recommender import clearance_range, recommend  # noqa: E402
from utils import is_peak_hour  # noqa: E402

app = FastAPI(title="SmartFlow API", version="1.0")

_MODELS = _ROOT / "models"
_clf = joblib.load(_MODELS / "severity_classifier.pkl")
_clo = joblib.load(_MODELS / "closure_predictor.pkl") if (_MODELS / "closure_predictor.pkl").exists() else None


class Event(BaseModel):
    event_cause: str = "vehicle_breakdown"
    zone: str = "Central Zone 1"
    hour_of_day: int = 9
    day_of_week: int = 0
    month: int = 6
    veh_type: str = "unknown"
    junction_repeat_count: int = 5
    corridor_7d_score: int = 5
    description: str = ""
    log: bool = True


@app.get("/health")
def health():
    return {"status": "ok", "closure_model": _clo is not None}


@app.post("/event")
def handle_event(ev: Event):
    sem = extract_semantic_type(ev.description)
    feats = {
        "cause_severity_weight":  CAUSE_SEVERITY_WEIGHT.get(ev.event_cause, 1),
        "road_closure_binary":    0,
        "event_semantic_encoded": _SEMANTIC_MAP.get(sem, 0),
        "hour_of_day":            ev.hour_of_day,
        "day_of_week":            ev.day_of_week,
        "month":                  ev.month,
        "is_peak_hour":           int(is_peak_hour(ev.hour_of_day)),
        "is_weekend":             int(ev.day_of_week >= 5),
        "junction_repeat_count":  ev.junction_repeat_count,
        "corridor_7d_score":      ev.corridor_7d_score,
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
    )

    if ev.log:
        log_decision({
            "event_cause": ev.event_cause, "zone": ev.zone, "hour_of_day": ev.hour_of_day,
            "predicted_severity": sev, "predicted_clearance_min": rec["estimated_clearance_minutes"],
            "recommended_personnel": rec["personnel_count"],
        }, project_root=_ROOT)

    return {
        "severity_triage": sev,
        "detected_event_type": sem,
        "road_closure_likelihood": closure_prob,
        "clearance_range": clearance_range(ev.event_cause),
        "recommendation": rec,
    }
